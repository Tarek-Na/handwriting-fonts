// Khatti live demo: the processing stepper, the upload area, and the result page's
// live typing in the writer's new font (rendered by the browser from /job/<id>/font.otf).
(function () {
  "use strict";
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  // ---- processing stepper: the request reads letters and draws forms (~3.5 s); the font is
  //      built afterwards and followed on the result page, so stage 3 starts there.
  function setStage(list, index) {
    $$(".stage", list).forEach((el, i) => {
      el.classList.toggle("done", i < index);
      el.classList.toggle("active", i === index);
      el.querySelector(".dot").innerHTML = i < index ? CHECK : String(i + 1);
    });
  }
  const CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" ' +
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
  $$("form[data-busy]").forEach((form) => form.addEventListener("submit", () => {
    const overlay = $("#overlay");
    if (!overlay) return;
    $("#overlay-title").textContent = form.dataset.busy;
    overlay.classList.add("on");
    const list = $("#overlay-stages");
    setStage(list, 0);
    setTimeout(() => setStage(list, 1), 1600);
    $$("button", form).forEach((b) => b.setAttribute("aria-disabled", "true"));
  }));

  // ---- upload area: preview the chosen photo, enable the button, accept a dropped file.
  const drop = $("#drop"), input = $("#photo"), go = $("#go");
  if (drop && input) {
    const show = (file) => {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (e) => { $("#thumb").src = e.target.result; drop.classList.add("has"); };
      reader.readAsDataURL(file);
      $("#chosen").textContent = "Photo chosen: " + file.name;
      go.disabled = false;
    };
    input.addEventListener("change", () => show(input.files[0]));
    ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; show(input.files[0]); }
    });
  }

  // ---- result page
  const page = $("#result");
  if (!page) return;
  const job = page.dataset.job, seed = page.dataset.seed, fallbackText = page.dataset.text;
  const text = $("#text"), live = $("#live");
  const family = "khatti-" + job + "-" + seed;
  let mode = "font", timer = null;

  function render() {
    const value = text.value.trim() || fallbackText;
    if (mode === "font") {
      live.textContent = value;
      return;
    }
    // The browser would not take the font: the server sets the text instead (as before).
    clearTimeout(timer);
    timer = setTimeout(() => {
      const url = "/job/" + job + "/sentence.png?text=" + encodeURIComponent(value) + "&v=" + seed;
      const img = new Image();
      img.alt = value;
      img.onload = () => { live.replaceChildren(img); };
      img.src = url;
    }, 300);
  }
  text.addEventListener("input", () => {
    $$(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
    render();
  });
  $$(".chip").forEach((chip) => chip.addEventListener("click", () => {
    text.value = chip.dataset.text;
    $$(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c === chip)));
    render();
    text.focus();
  }));

  function fontReady() {
    const face = new FontFace(family, "url(/job/" + job + "/font.otf?v=" + seed + ")");
    face.load().then((loaded) => {
      document.fonts.add(loaded);
      live.style.fontFamily = '"' + family + '"';
      live.classList.remove("fallback");
      mode = "font";
      render();
    }).catch(() => {
      live.classList.add("fallback");
      mode = "image";
      render();
    });
  }
  function reveal(state) {
    $("#building").classList.toggle("hidden", state !== "building");
    $("#failed").classList.toggle("hidden", state !== "error");
    $$(".when-ready").forEach((el) => el.classList.toggle("hidden", state !== "ready"));
    const stages = $("#font-stages");
    if (stages) setStage(stages, state === "ready" ? 3 : 2);
    if (state === "ready") {
      $("#status-line").textContent = "Your font is ready.";
      fontReady();
    }
  }
  function poll() {
    fetch("/job/" + job + "/status").then((r) => r.json()).then((s) => {
      if (s.font === "building") { setTimeout(poll, 800); return; }
      reveal(s.font);
    }).catch(() => setTimeout(poll, 1500));
  }
  reveal(page.dataset.font);
  if (page.dataset.font === "building") poll();
})();
