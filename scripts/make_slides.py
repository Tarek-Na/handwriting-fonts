"""The course presentation (CPEN426), built on the Slidesgo "Technology Infographics" template.

    python scripts/make_slides.py --template "Technology Infographics by Slidesgo (1).pptx" \\
        --out output/Khatti_CPEN426_presentation.pptx

Keeps the template's master, layouts and embedded Fira Sans fonts, removes every one of its
slides, and builds the talk with native shapes, a native table and a native chart in the
template's palette. Every number on a slide is taken from
``reports/Arabic variation model November 2026.md``. No image of handwriting is placed: where
one would help, a clearly marked placeholder box says what to insert. The .pptx and its
renders stay in ``output/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

# ------------------------------------------------------------------ the template's look
NAVY, DARK, ORANGE, ORANGE_D = "30475E", "222831", "F2A365", "DD915E"
GREY_L, GREY, GREY_D, WHITE = "EEEEEE", "CECECE", "606060", "FFFFFF"
HEAD, BODY, BODY_M = "Fira Sans Medium", "Fira Sans", "Fira Sans Medium"
ARABIC = "Arial"                                   # Fira Sans has no Arabic letters
#: The template's real palette written into its theme (its theme still held Google's stock one),
#: so native tables and charts in theme colours match the slides.
THEME_COLORS = {"dk1": DARK, "lt1": WHITE, "dk2": NAVY, "lt2": GREY_L, "accent1": NAVY, "accent2": ORANGE,
                "accent3": DARK, "accent4": GREY, "accent5": GREY_D, "accent6": ORANGE_D,
                "hlink": NAVY, "folHlink": GREY_D}

TITLE_ONLY, TITLE = "TITLE_ONLY", "TITLE"


def rgb(hexcolor: str) -> RGBColor:
    return RGBColor.from_string(hexcolor)


# ------------------------------------------------------------------ building blocks
def layout(prs, name):
    return next(l for l in prs.slide_layouts if l.name == name)


def set_run(run, size, color=DARK, font=BODY, bold=False, arabic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    face = ARABIC if arabic else font
    run.font.name = face
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rpr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rpr, qn(tag))
        el.set("typeface", face)
    if arabic:
        rpr.set("lang", "ar-SA")


def write(tf, paras, size=14, color=DARK, font=BODY, align=PP_ALIGN.LEFT, space_after=4, rtl=False):
    """Fill a text frame. ``paras``: list of paragraphs, each a str or a list of (text, options)."""
    tf.clear()
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        runs = [(para, {})] if isinstance(para, str) else para
        is_rtl = rtl or any(o.get("arabic") for _, o in runs)
        if is_rtl:
            p._p.get_or_add_pPr().set("rtl", "1")
        for text, opts in runs:
            r = p.add_run()
            r.text = text
            set_run(r, opts.get("size", size), opts.get("color", color), opts.get("font", font),
                    opts.get("bold", False), opts.get("arabic", False))
    return tf


def textbox(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, margin=0.05, name=None, **kw):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    write(tf, paras, **kw)
    if name:
        tb.name = name
    return tb


def shape(slide, kind, x, y, w, h, fill=None, line=None, line_w=1.25, name=None):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(line_w)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def shape_text(s, paras, size=12, color=WHITE, font=BODY_M, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, margin=0.06):
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    write(tf, paras, size=size, color=color, font=font, align=align, space_after=0)
    return s


def badge(slide, x, y, d, label, fill=NAVY, color=WHITE, size=14):
    """The template's motif: a number or short label in a filled circle."""
    return shape_text(shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill=fill, name=f"Badge {label}"), [label],
                      size=size, color=color)


def card(slide, x, y, w, h, title, body, accent=NAVY, number=None, body_size=12, name=None):
    """A rounded card: a light panel, a coloured heading, short body text."""
    panel = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill=WHITE, line=GREY, line_w=1, name=name)
    panel.adjustments[0] = 0.08
    tx = x + 0.15
    if number is not None:
        badge(slide, x + 0.15, y + 0.15, 0.42, str(number), fill=accent, size=13)
        tx = x + 0.68
    textbox(slide, tx, y + 0.12, w - (tx - x) - 0.12, 0.5, [[(title, {"font": HEAD, "color": accent, "size": 14})]],
            anchor=MSO_ANCHOR.MIDDLE, margin=0.02)
    textbox(slide, x + 0.15, y + 0.66, w - 0.3, h - 0.76, body if isinstance(body, list) else [body],
            size=body_size, color=GREY_D, margin=0.02)
    return panel


def insert_box(slide, x, y, w, h, what):
    """Where a handwriting image would help: a clearly marked box, left for the owners to fill locally."""
    s = shape(slide, MSO_SHAPE.RECTANGLE, x, y, w, h, fill=GREY_L, line=ORANGE_D, line_w=1.5, name=f"INSERT {what}")
    s.line.dash_style = 4                           # dash
    shape_text(s, [f"INSERT: {what}"], size=12, color=GREY_D, font=BODY_M)
    return s


def new_slide(prs, title=None, notes="", lay=TITLE_ONLY):
    slide = prs.slides.add_slide(layout(prs, lay))
    if title is not None and lay == TITLE_ONLY:
        t = slide.shapes.title
        t.left, t.top, t.width, t.height = Inches(0.78), Inches(0.38), Inches(8.45), Inches(0.62)
        write(t.text_frame, [title], size=26, color=DARK, font=HEAD, align=PP_ALIGN.CENTER, space_after=0)
        t.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def arrow(slide, x1, y1, x2, y2, color=GREY_D):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(1.75)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


# ------------------------------------------------------------------ the talk
def build(prs):
    # 1 ---- title
    s = prs.slides.add_slide(layout(prs, TITLE))
    ph = {p.placeholder_format.idx: p for p in s.placeholders}
    write(ph[0].text_frame, [[("Khatti", {"font": HEAD, "color": NAVY, "size": 44})],
                             [("Personal Arabic handwriting fonts with a writer-conditioned cVAE",
                               {"font": BODY_M, "color": DARK, "size": 22})]],
          align=PP_ALIGN.LEFT, space_after=6)
    ph[0].text_frame.vertical_anchor = MSO_ANCHOR.BOTTOM
    write(ph[1].text_frame, [[("Tarek Nasrallah · Antoine Lahoud", {"font": BODY_M, "color": DARK, "size": 16})],
                             [("Deep Learning (CPEN426) · 25/11/2026", {"color": GREY_D, "size": 14})]],
          align=PP_ALIGN.LEFT, space_after=2)
    for p_, (y, h) in ((ph[0], (0.9, 2.0)), (ph[1], (3.05, 0.95))):
        p_.left, p_.top, p_.width, p_.height = Inches(0.6), Inches(y), Inches(5.9), Inches(h)
    for d, x, y, fill in ((1.9, 7.3, 1.0, NAVY), (1.15, 6.55, 2.75, ORANGE), (0.8, 8.55, 3.15, DARK)):
        shape(s, MSO_SHAPE.OVAL, x, y, d, d, fill=fill, name="Title circle")
    shape_text(shape(s, MSO_SHAPE.OVAL, 7.55, 1.25, 1.4, 1.4, fill=WHITE, name="Title letter"),
               [[("خطّي", {"arabic": True, "color": NAVY, "size": 30, "bold": True})]], margin=0.0)
    s.notes_slide.notes_text_frame.text = (
        "TAREK (0:25). Good morning. We are Tarek Nasrallah and Antoine Lahoud. Our project is Khatti, "
        "which means 'my handwriting' in Arabic: from a single sheet of 20 handwritten letters, we build a "
        "working Arabic font in that person's handwriting. The deep-learning part is a writer-conditioned "
        "conditional VAE that draws the letter shapes the writer never wrote. Tarek takes the problem and the "
        "method; Antoine takes the evidence; then we show it live.")

    # 2 ---- the problem
    s = new_slide(prs, "Arabic letters change shape as they join", notes=(
        "TAREK (0:50). Arabic is written joined, right to left, and most letters have up to four shapes: "
        "on their own, at the start of a word, in the middle and at the end. You can see it for beh here. "
        "A font in someone's handwriting needs every one of those joined shapes, but asking a person to "
        "write all 66 of them is a lot. Our sheet asks for only 20 letters, each written once, on its own. "
        "So the joined forms, which the person never wrote, have to be generated: that is the job of the "
        "network. Placeholder: if we add a picture here, it shows joined forms of one writer."))
    forms = [("ب", "isolated"), ("بـ", "initial"), ("ـبـ", "medial"), ("ـب", "final")]
    for i, (glyph, name) in enumerate(forms):
        x = 6.45 - i * 1.62
        c = shape(s, MSO_SHAPE.OVAL, x, 1.3, 1.25, 1.25, fill=NAVY if i == 0 else WHITE, line=NAVY, line_w=2,
                  name=f"Form {name}")
        shape_text(c, [[(glyph, {"arabic": True, "size": 34, "color": WHITE if i == 0 else NAVY})]], margin=0.0)
        textbox(s, x - 0.2, 2.6, 1.65, 0.35, [name], size=13, color=GREY_D, align=PP_ALIGN.CENTER, font=BODY_M)
    textbox(s, 0.6, 3.15, 4.6, 1.9, [
        [("Up to 4 shapes per letter", {"font": HEAD, "color": NAVY, "size": 16})],
        "A handwriting font needs every joined shape",
        "Our sheet asks for 20 letters, each written once, alone",
        [("The network draws the joined forms the writer never wrote", {"font": BODY_M, "color": ORANGE_D})],
    ], size=14, space_after=6)
    insert_box(s, 5.45, 3.15, 3.95, 1.85, "example of joined forms")

    # 3 ---- why a distribution
    s = new_slide(prs, "We model versions, not one exact letter", notes=(
        "TAREK (0:45). Before building a model we checked what is predictable. A writer's letter changes "
        "every time they write it: the size of a form varies between one writer's own two copies about as "
        "much as the writer's other letters can predict it, a correlation of 0.28 against 0.37. And picking "
        "the training writer whose isolated letters look most like yours gives 0.441, barely above a random "
        "writer at 0.382. What is stable is the overall style. So the model's job is not one exact letter, "
        "but a distribution of plausible versions in the writer's style, and the font carries several "
        "versions of each letter."))
    for x, big, small, fill in ((1.0, "r = 0.28", "size, a writer's own two copies", ORANGE),
                                (3.85, "r = 0.37", "size, predicted from their other letters", NAVY)):
        c = shape(s, MSO_SHAPE.OVAL, x, 1.35, 2.1, 2.1, fill=fill, name=f"Stat {big}")
        shape_text(c, [[(big, {"size": 24, "font": HEAD})]], margin=0.0)
        textbox(s, x - 0.35, 3.55, 2.8, 0.6, [small], size=12, color=GREY_D, align=PP_ALIGN.CENTER)
    card(s, 6.55, 1.3, 2.95, 3.3, "So the network draws",
         ["several plausible versions of each joined form, in the writer's style",
          "Overall style is stable: isolated and joined size correlate at r = 0.78"], accent=ORANGE_D, body_size=13)
    textbox(s, 0.6, 4.4, 5.6, 0.6, [[("Choosing the most similar writer: 0.441 · a random writer: 0.382",
                                      {"color": GREY_D, "size": 12})]], align=PP_ALIGN.CENTER)

    # 4 ---- data
    s = new_slide(prs, "Data: AHAWP, split by writer", notes=(
        "TAREK (0:35). We train on AHAWP, a public Arabic handwriting dataset under CC BY 4.0: 82 writers, "
        "each writing 65 letter forms, 10 times. We split by writer, never by sample, so the model is always "
        "tested on people it has never seen: 49 writers for training, 17 for validation, where we made every "
        "choice, and 16 for the test, which we scored only once, at the end."))
    for i, (big, small) in enumerate((("82", "writers"), ("65", "letter forms"), ("10", "copies of each"))):
        x = 0.85 + i * 2.95
        c = shape(s, MSO_SHAPE.OVAL, x, 1.3, 1.45, 1.45, fill=NAVY if i != 1 else ORANGE, name=f"Data {small}")
        shape_text(c, [[(big, {"size": 32, "font": HEAD})]], margin=0.0)
        textbox(s, x + 1.5, 1.7, 1.35, 0.7, [small], size=14, color=DARK, font=BODY_M, anchor=MSO_ANCHOR.MIDDLE)
    y = 3.35
    total_w = 8.3
    for label, n, fill in (("train", 49, NAVY), ("val", 17, ORANGE), ("test", 16, DARK)):
        w = total_w * n / 82
        x0 = 0.85 + sum(total_w * m / 82 for _, m, _ in [("train", 49, 0), ("val", 17, 0)][: ["train", "val", "test"].index(label)])
        seg = shape(s, MSO_SHAPE.RECTANGLE, x0, y, w, 0.7, fill=fill, name=f"Split {label}")
        shape_text(seg, [[(f"{n} writers · {label}", {"size": 13})]])
    textbox(s, 0.85, 4.2, 8.3, 0.7, ["Choices made on the 17 validation writers; the 16 test writers scored once, at the end"],
            size=12, color=GREY_D, align=PP_ALIGN.CENTER)

    # 5 ---- method overview
    s = new_slide(prs, "Method: from 20 letters to a joining font", notes=(
        "TAREK (0:50). Here is the whole pipeline. The sheet gives us the writer's isolated letters. "
        "A style encoder reads them and pools them into one style vector. A conditional VAE, conditioned on "
        "that style and on which joined form we want, generates the form as a deformation of a skeleton "
        "template, several times, so we get several versions. Those drawings become the joined glyphs of an "
        "OpenType font, which joins letters and cycles through the versions as you type. The two blue boxes "
        "are the neural network; the next two slides open them up."))
    steps = [("20 isolated letters", "read from the photo", WHITE, NAVY),
             ("Style encoder", "attention over the letters", NAVY, WHITE),
             ("Conditional VAE", "over skeleton deformations", NAVY, WHITE),
             ("Joined forms", "several versions each", WHITE, NAVY),
             ("OpenType font", "joins and cycles versions", WHITE, NAVY)]
    xs = [0.35 + i * 1.9 for i in range(5)]
    nn = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, xs[1] - 0.12, 1.55, 3.64, 2.45, fill=None, line=ORANGE, line_w=1.5,
               name="Neural network group")
    nn.line.dash_style = 4
    textbox(s, xs[1] - 0.05, 1.6, 3.5, 0.3, [[("NEURAL NETWORK", {"font": HEAD, "color": ORANGE_D, "size": 11})]],
            align=PP_ALIGN.CENTER)
    for i, (head, sub, fill, txt) in enumerate(steps):
        b = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, xs[i], 2.05, 1.5, 1.6, fill=fill, line=NAVY, line_w=1.5,
                  name=f"Step {head}")
        b.adjustments[0] = 0.12
        shape_text(b, [[(head, {"font": HEAD, "size": 13, "color": txt})], [(sub, {"font": BODY, "size": 11, "color": txt})]])
        if i < 4:
            arrow(s, xs[i] + 1.52, 2.85, xs[i + 1] - 0.02, 2.85)
    textbox(s, 0.6, 4.3, 8.8, 0.7, [[("Novel: a writer-conditioned generative model over skeleton deformations, "
                                      "drawing several versions of each joined Arabic form", {"font": BODY_M, "color": DARK})]],
            size=14, align=PP_ALIGN.CENTER)

    # 6 ---- representation
    s = new_slide(prs, "A template skeleton plus a writer's moves", notes=(
        "TAREK (0:50). The network never sees pixels. For each of the 65 forms we take a template: the pen "
        "path of the most typical training writer. Any written copy is then described as that template's "
        "points moved onto the copy's skeleton, found with coherent point drift, an affine fit followed by "
        "smooth local moves. The moves are compressed with PCA to 24 numbers per form; 8 components already "
        "hold 99.5 percent of the variance. The table shows this representation is rich enough: fitted to a "
        "copy, it scores 0.689, higher than the writer's own second copy at 0.666; the template alone scores "
        "0.486. So the network's job becomes: produce 24 numbers for a form, in a writer's style."))
    for i, (head, body) in enumerate((("Template", "the medoid training writer's pen path, for each of 65 forms"),
                                      ("Moves", "coherent point drift: affine, then smooth local moves"),
                                      ("24 numbers", "PCA of the moves per form; 8 hold 99.5%"))):
        y = 1.25 + i * 1.22
        badge(s, 0.6, y + 0.15, 0.62, str(i + 1), fill=NAVY if i != 2 else ORANGE)
        textbox(s, 1.35, y, 3.3, 1.0, [[(head, {"font": HEAD, "color": NAVY, "size": 15})], body], size=12,
                color=GREY_D, space_after=2)
    rows = [("Drawn (val writers, joined forms)", "F1"),
            ("writer's own second copy", "0.666"),
            ("template + moves fitted to that copy", "0.689"),
            ("template + affine fit only", "0.619"),
            ("template alone, writer's pen", "0.486"),
            ("nearest training writer", "0.435")]
    table(s, 4.95, 1.3, 4.45, rows, col_w=(3.55, 0.9), highlight=2)
    textbox(s, 4.95, 4.35, 4.45, 0.5, ["Centreline F1 against the writer's typical copy"], size=11, color=GREY_D)

    # 7 ---- the model
    s = new_slide(prs, "The model: a writer-conditioned conditional VAE", notes=(
        "TAREK (0:55). The model has four parts. The style encoder embeds each isolated letter with its "
        "identity and pools them with attention into a 32-dimensional style vector; during training we hide "
        "30 percent of the letters at random so it cannot rely on any one letter. A conditional prior gives "
        "an 8-dimensional latent for a form and a style; the encoder is used only in training; and the decoder "
        "gives a Gaussian over the 24 numbers with a learned variance per component. That last detail "
        "matters: with a fixed variance the model collapsed and drew the same version every time, a spread "
        "of 0.01 against 0.3 in the data. Training uses all 10 copies of every form of the 49 training "
        "writers, 32,000 examples, 20,000 steps on a CPU."))
    parts = [("Style encoder", "attention pooling over the isolated letters → 32-d style; 30% letter dropout"),
             ("Conditional prior", "p(z | form, style), 8-d latent"),
             ("Encoder", "q(z | x, form, style), training only"),
             ("Decoder", "Gaussian over the 24 numbers, learned variance per component")]
    for i, (head, body) in enumerate(parts):
        x, y = 0.6 + (i % 2) * 3.0, 1.25 + (i // 2) * 1.75
        card(s, x, y, 2.8, 1.55, head, body, accent=NAVY if i != 3 else ORANGE_D, number=i + 1, body_size=12)
    c = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 6.75, 1.25, 2.65, 3.3, fill=NAVY, name="Collapse note")
    c.adjustments[0] = 0.08
    shape_text(c, [[("Why learned variance", {"font": HEAD, "size": 15, "color": ORANGE})],
                   [("A fixed variance collapsed: every version came out the same", {"size": 12})],
                   [("spread 0.01 against 0.3 in the data", {"size": 12, "color": ORANGE})],
                   [("Loss: Gaussian NLL + KL(q ‖ prior)", {"size": 12})]],
               align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, margin=0.18)

    # 8 ---- metrics
    s = new_slide(prs, "How we measure it", notes=(
        "ANTOINE (0:45). Thank you Tarek. To judge generated letters we compare them with the writer's real "
        "copies. For each joined form a method draws 5 versions, compared by aligned centreline F1, which "
        "measures how well two pen paths overlap. Realism: does each version match one of the writer's real "
        "copies? Coverage: is each real copy matched by some version, so the versions span how the person "
        "writes? And writer specificity: realism against this writer minus realism against another writer; "
        "only a method that knows the writer can score above zero. Every comparison is paired over writers, "
        "with 95 percent bootstrap intervals."))
    mets = [("Realism", "each version's best match among the writer's real copies", NAVY),
            ("Coverage", "each real copy's best match among the versions", ORANGE),
            ("Writer specificity", "realism for this writer minus realism for another; above 0 only if the method knows the writer", DARK)]
    for i, (head, body, col) in enumerate(mets):
        x = 0.6 + i * 3.0
        c = shape(s, MSO_SHAPE.OVAL, x + 0.95, 1.25, 0.9, 0.9, fill=col, name=f"Metric {head}")
        shape_text(c, [str(i + 1)], size=20)
        textbox(s, x, 2.25, 2.8, 1.6, [[(head, {"font": HEAD, "color": col if col != DARK else NAVY, "size": 15})], body],
                size=12, color=GREY_D, align=PP_ALIGN.CENTER, space_after=3)
    textbox(s, 0.6, 4.1, 8.8, 0.8, ["5 versions per joined form · aligned centreline F1 · paired over writers, 95% bootstrap intervals"],
            size=13, color=DARK, font=BODY_M, align=PP_ALIGN.CENTER)

    # 9 ---- main results
    s = new_slide(prs, "Results on 16 writers the model never saw", notes=(
        "ANTOINE (1:10). This is the main result, on the 16 test writers, scored once. Our model reaches "
        "0.605 realism, 0.518 coverage and plus 0.017 specificity. The fair baseline without a network is a "
        "Gaussian whose mean is predicted from the same isolated letters by ridge regression: 0.583, 0.506 "
        "and zero specificity. The network beats it on all three, and the paired intervals are all above "
        "zero: realism plus 0.021, better for 15 of the 16 writers. Against the plain Gaussian, realism is "
        "better for all 16 writers. The same network without the style encoder has almost no specificity, so "
        "that comes from reading the writer. The ceiling, the writer's own other copies, shows how much room "
        "is left."))
    rows = [("Test, 16 writers", "Realism", "Coverage", "Specificity"),
            ("Writer's own other copies (ceiling)", "0.713", "0.692", "+0.208"),
            ("cVAE + style encoder (ours)", "0.605", "0.518", "+0.017"),
            ("cVAE, no style", "0.592", "0.515", "+0.005"),
            ("Gaussian + ridge on the writer's letters", "0.583", "0.506", "−0.000"),
            ("Gaussian shape model", "0.572", "0.506", "−0.004")]
    table(s, 0.6, 1.2, 8.8, rows, col_w=(4.6, 1.4, 1.4, 1.4), highlight=2, muted=1)
    textbox(s, 0.6, 3.95, 8.8, 1.0, [
        [("Ours vs Gaussian + ridge:  ", {"font": BODY_M, "color": NAVY}),
         ("realism +0.021 [+0.016, +0.028], 15/16 writers · coverage +0.012 [+0.005, +0.019] · "
          "specificity +0.017 [+0.008, +0.029], 14/16", {})],
        [("Ours vs Gaussian shape model:  ", {"font": BODY_M, "color": NAVY}),
         ("realism +0.033 [+0.026, +0.040], 16/16 writers", {})]],
        size=12, color=DARK, space_after=4)

    # 10 ---- the style encoder is real
    s = new_slide(prs, "Evidence the style encoder reads the writer", notes=(
        "ANTOINE (1:10). Three checks, all on the validation writers, show the style encoder really reads "
        "who wrote the letters. First, the wrong-writer control: give the model another person's letters and "
        "its specificity flips from plus 0.033 to minus 0.031; its versions now match the other person. Own "
        "letters against another's: plus 0.064, better for all 17 writers. Second, few-shot: with a single "
        "letter the style is worse than none, but from 3 letters it works, so the demo refuses fewer than 3. "
        "Third, the chart: training on 12, 25 and 49 writers, specificity keeps rising and has not levelled "
        "off. That is direct evidence that the number of writers is the main limit."))
    card(s, 0.6, 1.2, 3.05, 1.65, "Wrong-writer control",
         ["Another writer's letters: specificity +0.033 → −0.031", "Own vs other: +0.064 [+0.030, +0.105], 17/17"],
         accent=NAVY, body_size=11)
    card(s, 0.6, 3.0, 3.05, 1.65, "Few-shot",
         ["1 letter: −0.005 · 3 letters: +0.029 · 18: +0.033", "So the demo needs at least 3 letters"],
         accent=ORANGE_D, body_size=11)
    data_curve(s, 3.85, 1.15, 5.6, 3.55)

    # 11 ---- robustness
    s = new_slide(prs, "Robust to real photos; small writing fixed", notes=(
        "ANTOINE (0:45). Before showing it live we stress-tested the whole photo pipeline: each validation "
        "writer's real letters written into the sheet, changed as another pen, phone or hand would, "
        "photographed, read back, and given to the model. A harsh photo, faint ink or a different pen made "
        "no difference. Only writing size mattered: letters at 0.6 times the size lost 0.060 realism. We now "
        "rescale writing outside the training writers' range before the model reads it, which cuts that "
        "loss to 0.028. Every box was read in every condition."))
    items = [("Harsh photo", "+0.000"), ("Faint ink", "+0.001"), ("Thin / thick pen", "+0.001 / −0.002"),
             ("Letters ×1.25", "−0.001")]
    for i, (what, val) in enumerate(items):
        y = 1.2 + i * 0.82
        badge(s, 0.6, y, 0.6, "✓", fill=NAVY, size=16)
        textbox(s, 1.35, y - 0.02, 3.4, 0.65, [[(what, {"font": BODY_M, "color": DARK, "size": 14})],
                                                [(f"realism {val}", {"color": GREY_D, "size": 12})]], space_after=0)
    c = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 5.0, 1.2, 4.4, 3.2, fill=WHITE, line=ORANGE, line_w=1.5, name="Size fix")
    c.adjustments[0] = 0.06
    textbox(s, 5.2, 1.3, 4.0, 0.5, [[("Only size mattered", {"font": HEAD, "color": ORANGE_D, "size": 16})]])
    for i, (big, small, col) in enumerate((("−0.060", "letters ×0.6, before", GREY_D), ("−0.028", "with the size fix", NAVY))):
        x = 5.25 + i * 2.05
        b = shape(s, MSO_SHAPE.OVAL, x + 0.3, 1.95, 1.3, 1.3, fill=col, name=f"Size {big}")
        shape_text(b, [[(big, {"size": 18, "font": HEAD})]], margin=0.0)
        textbox(s, x, 3.3, 1.9, 0.5, [small], size=12, color=GREY_D, align=PP_ALIGN.CENTER)
    textbox(s, 5.2, 3.8, 4.0, 0.5, ["Writing outside the training range (49–112 px) is rescaled first"],
            size=11, color=GREY_D, align=PP_ALIGN.CENTER)
    textbox(s, 0.6, 4.55, 8.8, 0.4, ["Realism change vs unchanged sheets, 17 validation writers; every box read in every condition"],
            size=11, color=GREY_D, align=PP_ALIGN.CENTER)

    # 12 ---- live demo
    s = new_slide(prs, "Live demo", notes=(
        "BOTH (0:15 on the slide, then about 4:00 live). TAREK hands over to the browser at "
        "http://localhost:8000 (demo started before the talk; it warms up for about 30 s at launch). "
        "Live, about 4 minutes: "
        "1) The instructor's sheet, filled in before the talk: photograph it and upload it, typing his name "
        "as the font name. "
        "2) The page shows his 20 letters as read (about 5 s) and, as the centrepiece, the network's drawings: "
        "joined forms he never wrote, four versions each. Point out that each version differs, as a hand does. "
        "3) When the font is ready (about 15 s more), type a sentence in the box, or click a sample sentence: "
        "it is set by the font itself. "
        "4) Download the .otf; optionally click 'Draw new versions'. "
        "5) If time allows, open 'How it works'. "
        "FALLBACK if the photo fails (no squares found, poor light): click 'Try it with a sample sheet' on "
        "the home page and say it is a writer the model never saw."))
    for i, (head, body, icon) in enumerate((("Print the sheet", "20 boxes, one per letter", "1"),
                                            ("Write each letter once", "on its own, on the red line", "2"),
                                            ("Photograph and upload", "all four squares in view", "3"))):
        x = 0.85 + i * 2.95
        b = shape(s, MSO_SHAPE.OVAL, x + 0.55, 1.35, 1.5, 1.5, fill=NAVY if i != 1 else ORANGE, name=f"Demo step {icon}")
        shape_text(b, [[(icon, {"size": 36, "font": HEAD})]], margin=0.0)
        textbox(s, x, 3.0, 2.6, 0.9, [[(head, {"font": HEAD, "color": NAVY, "size": 15})], body],
                size=12, color=GREY_D, align=PP_ALIGN.CENTER, space_after=2)
        if i < 2:
            arrow(s, x + 2.15, 2.1, x + 3.4, 2.1)
    url = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 2.9, 4.1, 4.2, 0.65, fill=DARK, name="Demo address")
    url.adjustments[0] = 0.5
    shape_text(url, [[("http://localhost:8000", {"font": BODY_M, "size": 18, "color": ORANGE})]])

    # 13 ---- limits
    s = new_slide(prs, "Limits", notes=(
        "ANTOINE (0:45). We want to be clear about the limits. The gains are real but modest: on the test "
        "writers our specificity is plus 0.017, while a writer's own copies reach plus 0.208, so we capture "
        "only part of a person's style. Style that lives in the connecting strokes, such as a writer's long "
        "joins, is not reproduced, because isolated letters have no connections to learn from. At word "
        "length, even a writer's own repeated words differ more than writers do, so we make no word-level "
        "claim. And the data curve says the main limit is the number of writers: 49."))
    lim = [("Modest gains", "specificity +0.017 against +0.208 for the writer's own copies"),
           ("Connecting strokes", "style in the joins is not reproduced: isolated letters have none"),
           ("Letter-level claims only", "a writer's own repeated words differ more than writers do"),
           ("Data is the main limit", "49 training writers; specificity still rising at 49")]
    for i, (head, body) in enumerate(lim):
        x, y = 0.6 + (i % 2) * 4.5, 1.25 + (i // 2) * 1.75
        card(s, x, y, 4.3, 1.55, head, body, accent=NAVY if i % 3 == 0 else ORANGE_D, number=i + 1, body_size=12)

    # 14 ---- future work
    s = new_slide(prs, "Future work", notes=(
        "ANTOINE (0:40). Four directions. More paired data first, since that is the measured limit: a dataset "
        "such as HACDB, with 50 more writers, would roughly double ours. A pretrained writer-identification "
        "backbone for the style encoder, with care: in our tests, training the encoder to tell writers apart "
        "lowered specificity, because what identifies a person is not what the generator needs. A diffusion "
        "or flow-matching generator in the same 24-dimensional space. And a larger, stroke-level model that "
        "can learn the connecting strokes."))
    fut = [("More paired data", "more writers with isolated and joined forms"),
           ("Writer-ID backbone", "pretrained style encoder; telling writers apart alone lowered specificity"),
           ("Diffusion or flow matching", "a different generator in the same 24-number space"),
           ("Stroke-level model", "learn the connecting strokes, not only letter shapes")]
    for i, (head, body) in enumerate(fut):
        x = 0.6 + i * 2.25
        b = shape(s, MSO_SHAPE.HEXAGON, x + 0.55, 1.6, 1.1, 0.95, fill=NAVY if i % 2 == 0 else ORANGE,
                  name=f"Future {i + 1}")
        shape_text(b, [str(i + 1)], size=22, font=HEAD)
        textbox(s, x, 2.7, 2.1, 2.2, [[(head, {"font": HEAD, "color": NAVY, "size": 14})], body],
                size=12, color=GREY_D, align=PP_ALIGN.CENTER, space_after=3)

    # 15 ---- thank you
    s = prs.slides.add_slide(layout(prs, "SECTION_HEADER"))
    t = s.shapes.title
    write(t.text_frame, [[("Thank you", {"font": HEAD, "color": NAVY, "size": 40})],
                         [("Questions?", {"font": BODY_M, "color": ORANGE_D, "size": 24})]],
          align=PP_ALIGN.CENTER, space_after=4)
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(1.85), Inches(8.8), Inches(1.6)
    shape_text(shape(s, MSO_SHAPE.OVAL, 4.4, 0.55, 1.2, 1.2, fill=NAVY, name="Thanks letter"),
               [[("خطّي", {"arabic": True, "color": WHITE, "size": 22, "bold": True})]], margin=0.0)
    textbox(s, 0.6, 3.6, 8.8, 0.45, ["Tarek Nasrallah · Antoine Lahoud · Deep Learning (CPEN426)"],
            size=13, color=GREY_D, align=PP_ALIGN.CENTER)
    credit = textbox(s, 0.6, 4.75, 8.8, 0.35, [[("Presentation template by ", {}), ("Slidesgo", {}),
                                                (" · infographics by ", {}), ("Freepik", {})]],
                     size=9, color=GREY_D, align=PP_ALIGN.CENTER, name="Template credit")
    runs = credit.text_frame.paragraphs[0].runs
    runs[1].hyperlink.address = "https://slidesgo.com/"
    runs[3].hyperlink.address = "https://www.freepik.com/"
    s.notes_slide.notes_text_frame.text = (
        "BOTH (0:10). Thank you. We are happy to take questions. "
        "(The template's licence asks for the Slidesgo and Freepik credit kept on this slide.)")


def table(slide, x, y, w, rows, col_w, highlight=None, muted=None):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(0.42 * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(0.42)
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.theme_color = MSO_THEME_COLOR.ACCENT_1
                color, font = WHITE, BODY_M
            elif i == highlight:
                cell.fill.fore_color.theme_color = MSO_THEME_COLOR.ACCENT_2
                color, font = DARK, HEAD
            else:
                cell.fill.fore_color.rgb = rgb(WHITE if i % 2 else "F4F4F4")
                color, font = (GREY_D if i == muted else DARK), BODY
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            write(cell.text_frame, [val], size=12, color=color, font=font,
                  align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER, space_after=0)
    tbl = t._tbl
    tbl_pr = tbl.find(qn("a:tblPr"))
    tbl_pr.set("firstRow", "1")
    tbl_pr.set("bandRow", "0")
    style = tbl_pr.find(qn("a:tableStyleId"))
    if style is not None:
        tbl_pr.remove(style)
    return t


def data_curve(slide, x, y, w, h):
    """Writer specificity of models trained on 12, 25 and 49 writers (report 5d), both seeds."""
    data = CategoryChartData(number_format='+0.000;-0.000')
    data.categories = ["12 writers", "25 writers", "49 writers"]
    data.add_series("seed 0", (0.008, 0.018, 0.033))
    data.add_series("seed 1", (0.008, 0.022, None))
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), data)
    ch = gf.chart
    ch.has_title = True
    ch.chart_title.text_frame.text = "Specificity keeps rising with more training writers"
    for r in ch.chart_title.text_frame.paragraphs[0].runs:
        set_run(r, 13, DARK, BODY_M)
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(11)
    ch.legend.font.name = BODY
    for i, series in enumerate(ch.series):
        col = MSO_THEME_COLOR.ACCENT_1 if i == 0 else MSO_THEME_COLOR.ACCENT_2
        series.format.line.color.theme_color = col
        series.format.line.width = Pt(2.5)
        series.marker.style = XL_MARKER_STYLE.CIRCLE
        series.marker.size = 9
        series.marker.format.fill.solid()
        series.marker.format.fill.fore_color.theme_color = col
        series.marker.format.line.color.theme_color = col
        series.smooth = False
        series.data_labels.show_value = True
        series.data_labels.number_format = '+0.000'
        series.data_labels.number_format_is_linked = False
        series.data_labels.font.size = Pt(10)
        series.data_labels.font.name = BODY
        series.data_labels.font.color.rgb = rgb(DARK)
    from pptx.enum.chart import XL_LABEL_POSITION
    ch.series[0].data_labels.position = XL_LABEL_POSITION.BELOW      # seed 0 is the lower line at 25
    ch.series[1].data_labels.position = XL_LABEL_POSITION.ABOVE
    va = ch.value_axis
    va.minimum_scale, va.maximum_scale = 0.0, 0.04
    va.major_unit = 0.01
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(GREY_L)
    va.tick_labels.font.size = Pt(10)
    va.tick_labels.font.name = BODY
    va.tick_labels.font.color.rgb = rgb(GREY_D)
    va.tick_labels.number_format = '+0.00;-0.00'
    va.tick_labels.number_format_is_linked = False
    va.format.line.fill.background()
    ca = ch.category_axis
    ca.tick_labels.font.size = Pt(11)
    ca.tick_labels.font.name = BODY
    ca.tick_labels.font.color.rgb = rgb(DARK)
    ca.format.line.color.rgb = rgb(GREY)
    return gf


# ------------------------------------------------------------------ the file
def strip_template_slides(prs):
    ids = prs.slides._sldIdLst
    for sid in list(ids):
        prs.part.drop_rel(sid.rId)
        ids.remove(sid)


def set_properties(prs):
    cp = prs.core_properties
    cp.title = "Khatti: Personal Arabic Handwriting Fonts"
    cp.author = "Tarek Nasrallah, Antoine Lahoud"
    cp.last_modified_by = "Tarek Nasrallah"
    cp.subject = "Deep Learning (CPEN426) course project"
    cp.keywords = ""
    cp.comments = ""
    cp.category = ""
    cp.identifier = ""
    cp.content_status = ""
    cp.version = ""
    cp.revision = 1


def finish_package(path: Path):
    """Write the template's real palette and Fira Sans into its theme; clear the app properties."""
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "deck.pptx"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if re.match(r"ppt/theme/theme\d+\.xml", item.filename):
                x = data.decode("utf8")
                for name, val in THEME_COLORS.items():
                    x = re.sub(rf"(<a:{name}>)<a:(?:srgbClr|sysClr)[^>]*?(?:/>|>.*?</a:(?:srgbClr|sysClr)>)",
                               rf'\1<a:srgbClr val="{val}"/>', x, count=1, flags=re.S)
                x = re.sub(r'(<a:majorFont><a:latin typeface=")[^"]*', r"\1" + HEAD, x)
                x = re.sub(r'(<a:minorFont><a:latin typeface=")[^"]*', r"\1" + BODY, x)
                data = x.encode("utf8")
            if re.match(r"ppt/(theme/theme|slideMasters/slideMaster|slideLayouts/slideLayout|notesMasters/notesMaster)\d+\.xml",
                        item.filename):
                x = data.decode("utf8")
                x = re.sub(r'(<a:theme [^>]*?name=")[^"]*', lambda m: m.group(1) + "Khatti", x)  # the template's theme name
                x = re.sub(r'name="Google Shape;(\d+);[^"]*"', lambda m: f'name="Shape {m.group(1)}"', x)  # Google Slides export names
                data = x.encode("utf8")
            if item.filename == "docProps/app.xml":
                x = data.decode("utf8")
                x = re.sub(r"<Company>.*?</Company>", "<Company></Company>", x, flags=re.S)
                x = re.sub(r"<Manager>.*?</Manager>", "<Manager></Manager>", x, flags=re.S)
                data = x.encode("utf8")
            zout.writestr(item, data)
    shutil.move(str(out), str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--template", type=Path, default=Path("Technology Infographics by Slidesgo (1).pptx"))
    ap.add_argument("--out", type=Path, default=Path("output/Khatti_CPEN426_presentation.pptx"))
    args = ap.parse_args()
    if "output" not in args.out.parts:
        raise SystemExit("the deck stays under output/ (git-ignored)")
    prs = Presentation(str(args.template))
    strip_template_slides(prs)
    build(prs)
    set_properties(prs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(args.out))
    finish_package(args.out)
    print("->", args.out, f"({len(Presentation(str(args.out)).slides)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
