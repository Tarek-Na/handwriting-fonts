"""Phase 1 training loop.

Written around the fact that training happens on a free Colab GPU, which can be
reclaimed at any moment. Every design choice below that looks over-careful is
paying for that:

* checkpoints are atomic and written on a wall-clock interval, not only at
  epoch boundaries, so an eviction costs minutes rather than hours;
* resume restores the optimizer, EMA, scaler, step count and RNG state, so a
  resumed run is a continuation rather than a warm restart;
* the sample grid is written every eval, because looking at glyphs is the only
  honest way to judge this task and a loss curve will happily keep dropping
  while the output is unusable.

The adversarial terms default to off. Get reconstruction working, confirm the
letterforms are right, then enable them to harden the stroke edges.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.dataset import DatasetConfig, FontEvalDataset, GlyphPairDataset, GlyphStore
from .models.discriminator import (
    DiscriminatorConfig,
    PatchDiscriminator,
    hinge_d_loss,
    hinge_g_loss,
)
from .models.generator import GeneratorConfig, GlyphGenerator
from .models.losses import GlyphLoss, LossConfig, pixel_metrics, style_contrastive
from .utils import (
    CSVLogger,
    EMA,
    atomic_save,
    describe_device,
    pick_device,
    save_image_grid,
    set_seed,
    to_json,
    write_json,
)

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    cache_dir: str = "data/cache"
    out_dir: str = "runs/phase1"

    steps: int = 120_000
    batch_size: int = 32
    lr: float = 2e-4
    lr_disc: float = 1e-4
    #: 0.9 for reconstruction. Drop beta1 to 0.5 (the usual GAN setting) if the
    #: adversarial terms are switched on.
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    warmup_steps: int = 1_000
    min_lr_ratio: float = 0.05
    grad_clip: float = 5.0

    ema_decay: float = 0.999
    #: Enable the discriminator after this many steps. Reconstruction should be
    #: working before adversarial pressure is applied.
    adversarial_start: int = 40_000

    num_workers: int = 4
    seed: int = 0
    amp: bool = True
    device: str = "auto"
    #: Initialize generator weights (not optimizer state or step count) from
    #: another run's checkpoint. Ignored when this run has its own checkpoint to
    #: resume. Letterform structure transfers between runs even when framing or
    #: losses change, and it is the slowest thing to learn from scratch.
    init_from: str | None = None

    log_every: int = 100
    eval_every: int = 2_000
    sample_every: int = 2_000
    checkpoint_every_steps: int = 2_000
    checkpoint_every_seconds: float = 600.0
    #: Keep a separate step{N}.pt at these steps. last.pt is overwritten every
    #: checkpoint, so a staged change mid-run (switching the adversarial terms
    #: on, say) would otherwise leave no model from just before it.
    snapshot_steps: tuple[int, ...] = ()
    eval_batches: int = 24

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DatasetConfig = field(default_factory=DatasetConfig)


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base LR."""
    if step < cfg.warmup_steps:
        return (step + 1) / max(cfg.warmup_steps, 1)
    import math

    progress = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = pick_device(cfg.device)
        set_seed(cfg.seed)

        log.info("device: %s", describe_device(self.device))

        self.store = GlyphStore(cfg.cache_dir)
        train_ds = GlyphPairDataset(
            self.store, replace(cfg.data, split="train"), seed=cfg.seed
        )
        self.n_chars = len(self.store.charset)
        self.n_fonts = len(train_ds.font_ids)
        log.info(
            "corpus: %d train fonts, %d glyphs, image %dpx",
            self.n_fonts, self.n_chars, self.store.image_size,
        )

        # The charset and font count are properties of the cache, not knobs, so
        # they are pushed into the model configs rather than trusted from YAML.
        cfg.generator = replace(
            cfg.generator, n_chars=self.n_chars, image_size=self.store.image_size
        )
        cfg.discriminator = replace(
            cfg.discriminator,
            n_chars=self.n_chars,
            n_fonts=self.n_fonts,
            image_size=self.store.image_size,
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
        )

        eval_ds = FontEvalDataset(self.store, split="val", ref_seed="seed24", max_fonts=40)
        self.eval_loader = DataLoader(
            eval_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
        )
        log.info("val set: %d glyphs from %d fonts", len(eval_ds), len(eval_ds.records))

        self.generator = GlyphGenerator(cfg.generator).to(self.device)
        self.discriminator = PatchDiscriminator(cfg.discriminator).to(self.device)
        log.info("generator: %.2fM params", self.generator.num_parameters() / 1e6)

        self.opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay,
        )
        self.opt_d = torch.optim.AdamW(
            self.discriminator.parameters(), lr=cfg.lr_disc, betas=cfg.betas
        )

        self.ema = EMA(self.generator, cfg.ema_decay)
        self.criterion = GlyphLoss(cfg.loss)
        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.step = 0
        self.best_iou = -1.0
        self.logger = CSVLogger(self.out_dir / "metrics.csv")
        write_json(self.out_dir / "config.json", to_json(cfg))

        self._sample_batch = None

    # -- checkpointing ------------------------------------------------------
    @property
    def _ckpt_path(self) -> Path:
        return self.out_dir / "last.pt"

    def save_checkpoint(self, name: str = "last.pt") -> None:
        atomic_save(
            {
                "step": self.step,
                "best_iou": self.best_iou,
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "ema": self.ema.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": to_json(self.cfg),
                "charset": self.store.charset.name,
                "rng": torch.get_rng_state(),
            },
            self.out_dir / name,
        )

    def maybe_resume(self) -> bool:
        if not self._ckpt_path.is_file():
            return False
        state = torch.load(self._ckpt_path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.ema.load_state_dict(state["ema"])
        self.scaler.load_state_dict(state["scaler"])
        self.step = state["step"]
        self.best_iou = state.get("best_iou", -1.0)
        if "rng" in state:
            torch.set_rng_state(state["rng"].cpu().to(torch.uint8))
        log.info("resumed from step %d", self.step)
        return True

    # -- steps --------------------------------------------------------------
    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _set_lr(self) -> float:
        mult = cosine_lr(self.step, self.cfg)
        for group in self.opt_g.param_groups:
            group["lr"] = self.cfg.lr * mult
        for group in self.opt_d.param_groups:
            group["lr"] = self.cfg.lr_disc * mult
        return self.cfg.lr * mult

    def _style_views(self, batch: dict):
        """Encode each sample's references twice, from disjoint halves.

        A positive pair is one hand seen through two different sets of its own
        letters. Splitting rather than re-sampling keeps the cost flat: the
        style backbone still sees every reference exactly once.
        """
        mask = batch["ref_mask"]
        counts = mask.sum(dim=1)
        half = counts // 2

        index = torch.arange(mask.shape[1], device=mask.device)[None, :]
        rank = (mask.cumsum(dim=1) - 1)
        first = mask & (rank < half[:, None])
        second = mask & (rank >= half[:, None]) & (index >= 0)

        refs = batch["refs"]
        view_a = self.generator.encode_style(refs, first)
        view_b = self.generator.encode_style(refs, second)
        return view_a, view_b, counts >= 2

    @property
    def adversarial_active(self) -> bool:
        return (
            self.cfg.loss.adversarial > 0.0 and self.step >= self.cfg.adversarial_start
        )

    def train_step(self, batch: dict) -> dict[str, float]:
        cfg = self.cfg
        batch = self._to_device(batch)
        lr = self._set_lr()

        amp_ctx = torch.amp.autocast("cuda", enabled=self.use_amp)

        # --- discriminator -------------------------------------------------
        d_stats: dict[str, float] = {}
        if self.adversarial_active:
            with amp_ctx:
                with torch.no_grad():
                    fake = self.generator(
                        batch["content"], batch["refs"], batch["ref_mask"], batch["char_id"]
                    )["image"]
                real_out = self.discriminator(batch["target"])
                fake_out = self.discriminator(fake)
                d_loss = hinge_d_loss(real_out["patch"], fake_out["patch"])
                if cfg.loss.char_aux > 0:
                    d_loss = d_loss + cfg.loss.char_aux * F.cross_entropy(
                        real_out["char_logits"], batch["char_id"]
                    )
                if cfg.loss.font_aux > 0 and "font_logits" in real_out and "font_id" in batch:
                    d_loss = d_loss + cfg.loss.font_aux * F.cross_entropy(
                        real_out["font_logits"], batch["font_id"]
                    )
            self.opt_d.zero_grad(set_to_none=True)
            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.opt_d)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), cfg.grad_clip)
            self.scaler.step(self.opt_d)
            d_stats["d_loss"] = float(d_loss.detach())

        # --- generator -----------------------------------------------------
        with amp_ctx:
            out = self.generator(
                batch["content"], batch["refs"], batch["ref_mask"], batch["char_id"]
            )
            g_loss, parts = self.criterion.reconstruction(
                out["image"], batch["target"], out.get("advance"), batch.get("advance"),
                target_weight=batch.get("target_weight"),
            )

            if cfg.loss.style_contrastive > 0:
                view_a, view_b, has_two = self._style_views(batch)
                contrast = style_contrastive(
                    view_a, view_b, has_two, cfg.loss.style_temperature
                )
                g_loss = g_loss + cfg.loss.style_contrastive * contrast
                parts["style_con"] = float(contrast.detach())
            if self.adversarial_active:
                fake_out = self.discriminator(out["image"])
                adv = hinge_g_loss(fake_out["patch"])
                g_loss = g_loss + cfg.loss.adversarial * adv
                parts["adv"] = float(adv.detach())
                if cfg.loss.char_aux > 0:
                    char_ce = F.cross_entropy(fake_out["char_logits"], batch["char_id"])
                    g_loss = g_loss + cfg.loss.char_aux * char_ce
                    parts["char_ce"] = float(char_ce.detach())

        self.opt_g.zero_grad(set_to_none=True)
        self.scaler.scale(g_loss).backward()
        self.scaler.unscale_(self.opt_g)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), cfg.grad_clip)
        self.scaler.step(self.opt_g)
        self.scaler.update()
        self.ema.update(self.generator)

        parts.update(d_stats)
        parts["g_loss"] = float(g_loss.detach())
        parts["lr"] = lr
        parts["grad_norm"] = float(grad_norm)
        return parts

    # -- evaluation ---------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, use_ema: bool = True) -> dict[str, float]:
        self.generator.eval()
        backup = self.ema.copy_to(self.generator) if use_ema else None

        totals: dict[str, float] = {}
        n = 0
        for i, batch in enumerate(self.eval_loader):
            if i >= self.cfg.eval_batches:
                break
            batch = self._to_device(batch)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.generator(
                    batch["content"], batch["refs"], batch["ref_mask"], batch["char_id"]
                )
            metrics = pixel_metrics(out["image"].float(), batch["target"])
            if "advance" in out and "advance" in batch:
                metrics["advance_l1"] = float(
                    (out["advance"].float() - batch["advance"]).abs().mean()
                )
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1

        if backup is not None:
            self.ema.restore(self.generator, backup)
        self.generator.train()
        return {f"val_{k}": v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def write_samples(self) -> None:
        """Save a content / generated / target strip for eyeballing."""
        if self._sample_batch is None:
            self._sample_batch = self._to_device(next(iter(self.eval_loader)))
        batch = self._sample_batch

        self.generator.eval()
        backup = self.ema.copy_to(self.generator)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            out = self.generator(
                batch["content"], batch["refs"], batch["ref_mask"], batch["char_id"]
            )
        self.ema.restore(self.generator, backup)
        self.generator.train()

        n = min(16, batch["content"].shape[0])
        # Rows read: what we asked for / what we got / what was correct.
        strip = torch.cat(
            [batch["content"][:n], out["image"][:n].float(), batch["target"][:n]]
        )
        save_image_grid(self.out_dir / "samples" / f"step{self.step:07d}.png", strip, n_cols=n)

    # -- main loop ----------------------------------------------------------
    def warm_start(self, path: str) -> None:
        """Load generator + EMA weights only; optimizer and schedule start fresh."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(state["generator"])
        if "ema" in state:
            self.ema.load_state_dict(state["ema"])
        else:
            self.ema = EMA(self.generator, self.cfg.ema_decay)
        log.info("warm-started generator from %s (its step %s)", path, state.get("step"))

    def train(self) -> None:
        cfg = self.cfg
        if not self.maybe_resume() and cfg.init_from:
            self.warm_start(cfg.init_from)
        self.generator.train()
        self.discriminator.train()

        last_ckpt = time.time()
        running: dict[str, float] = {}
        t0 = time.time()

        loader = iter(self.train_loader)
        while self.step < cfg.steps:
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(self.train_loader)
                batch = next(loader)

            stats = self.train_step(batch)
            self.step += 1
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v

            if self.step % cfg.log_every == 0:
                avg = {k: v / cfg.log_every for k, v in running.items()}
                running.clear()
                rate = cfg.log_every * cfg.batch_size / (time.time() - t0)
                t0 = time.time()
                log.info(
                    "step %6d/%d | l1 %.4f dice %.4f g %.4f | %.0f img/s",
                    self.step, cfg.steps, avg.get("l1", 0), avg.get("dice", 0),
                    avg.get("g_loss", 0), rate,
                )
                self.logger.log({"step": self.step, "img_per_s": rate, **avg})

            if self.step % cfg.eval_every == 0:
                metrics = self.evaluate()
                log.info(
                    "  eval @ %d | IoU %.4f  L1 %.4f  coverage %.3f",
                    self.step, metrics["val_iou"], metrics["val_l1"],
                    metrics.get("val_coverage_ratio", 0),
                )
                self.logger.log({"step": self.step, **metrics})
                if metrics["val_iou"] > self.best_iou:
                    self.best_iou = metrics["val_iou"]
                    self.save_checkpoint("best.pt")
                    log.info("  new best IoU %.4f", self.best_iou)

            if self.step % cfg.sample_every == 0:
                self.write_samples()

            due = (
                self.step % cfg.checkpoint_every_steps == 0
                or time.time() - last_ckpt > cfg.checkpoint_every_seconds
            )
            if due:
                self.save_checkpoint()
                last_ckpt = time.time()

            if self.step in cfg.snapshot_steps:
                self.save_checkpoint(f"step{self.step:06d}.pt")
                log.info("  snapshot saved at step %d", self.step)

        self.save_checkpoint()
        log.info("training complete at step %d (best val IoU %.4f)", self.step, self.best_iou)
