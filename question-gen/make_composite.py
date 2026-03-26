"""
Generate official PAT-format composite images from individual view PNGs.

For each question directory, creates composite.png showing:
  Left panel  — TOP VIEW (upper-left quadrant), ? FRONT VIEW (lower-left),
                END VIEW (lower-right), with orthographic reference cross
  Right panel — Answer choices A B C D, each with its own reference cross,
                image in the lower-left quadrant (matching PAT exam layout)

Usage (called from config.py after prepare_selected_questions):
    from make_composite import make_composite_for_all
    make_composite_for_all(selected_questions_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image, ImageDraw, ImageFont

# ── Layout constants ──────────────────────────────────────────────────────────
VIEW      = 150    # px — side length of each individual view tile
PAD       = 14     # px — gap between a tile edge and the cross centre
LABEL_H   = 44     # px — height of the label row beneath the cross area (fits two stacked lines)
LW        = 2      # px — axis line width
BG        = (255, 255, 255)
FG        = (0, 0, 0)

# Question-panel geometry
# Cross centre at (qcx, qcy); upper-left = TOP VIEW, lower-right = END VIEW
Q_CROSS_W = 2 * VIEW + 2 * PAD   # total cross area width
Q_CROSS_H = 2 * VIEW + 2 * PAD   # total cross area height
Q_W       = Q_CROSS_W
Q_H       = Q_CROSS_H + LABEL_H  # cross area + label row

# The cross centre in the question panel
QCX = VIEW + PAD
QCY = VIEW + PAD

# Each choice panel is sized so its cross area matches Q_CROSS_H in height,
# giving the same horizontal reference line across the whole composite.
# Cross centre at (ccx, QCY) for each panel — same vertical position as question.
CH_W      = VIEW + 2 * PAD        # width of one choice panel
TOTAL_H   = Q_H                   # all panels share this height
TOTAL_W   = Q_W + 20 + 4 * CH_W  # 20px gap between question and choices

FONT_SIZE       = 15
LABEL_FONT_SIZE = 14
QM_FONT_SIZE    = 36


def _load(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((size, size), Image.LANCZOS)


def _font(size: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def _centred(draw: ImageDraw.Draw, text: str, cx: int, y: int,
             fnt: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]) -> None:
    bbox = draw.textbbox((0, 0), text, font=fnt)
    draw.text((cx - (bbox[2] - bbox[0]) // 2, y), text, fill=FG, font=fnt)


def make_composite(q_dir: Path, out_path: Path) -> None:
    """
    Create a single PAT-format composite PNG for one question directory.
    q_dir must contain:
        input/top_view.png  input/end_view.png
        answers/A.png  answers/B.png  answers/C.png  answers/D.png
    """
    top_view = _load(q_dir / "input"   / "top_view.png", VIEW)
    end_view = _load(q_dir / "input"   / "end_view.png", VIEW)
    choices  = {c: _load(q_dir / "answers" / f"{c}.png", VIEW) for c in "ABCD"}

    lf  = _font(LABEL_FONT_SIZE)
    qmf = _font(QM_FONT_SIZE)

    canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), BG)

    # ── Paste all images first ────────────────────────────────────────────────
    # TOP VIEW — upper-left quadrant
    canvas.paste(top_view, (QCX - VIEW, QCY - VIEW))
    # END VIEW — lower-right quadrant
    canvas.paste(end_view, (QCX, QCY))

    gap = 20
    choice_positions = []
    for i, letter in enumerate("ABCD"):
        x0  = Q_W + gap + i * CH_W
        ccx = x0 + VIEW
        ccy = QCY
        canvas.paste(choices[letter], (ccx - VIEW, ccy))
        choice_positions.append((x0, ccx, ccy))

    # ── Draw all lines on top of images ──────────────────────────────────────
    draw = ImageDraw.Draw(canvas)

    # Question panel axis lines
    draw.line([(QCX, 0),       (QCX, Q_CROSS_H)],  fill=FG, width=LW)  # vertical
    draw.line([(0,   QCY),     (Q_CROSS_W, QCY)],  fill=FG, width=LW)  # horizontal

    # Choice panel axis lines
    for x0, ccx, ccy in choice_positions:
        draw.line([(ccx, 0),       (ccx, Q_CROSS_H)],      fill=FG, width=LW)
        draw.line([(x0,  ccy),     (x0 + CH_W, ccy)],      fill=FG, width=LW)

    # ── Text labels ───────────────────────────────────────────────────────────
    # "?" in lower-left quadrant of question panel
    ql_cx = QCX // 2
    ql_cy = QCY + (Q_CROSS_H - QCY) // 2
    _centred(draw, "?", ql_cx, ql_cy - QM_FONT_SIZE // 2, qmf)

    # Labels below the cross area
    line1_y  = Q_CROSS_H + 4
    line2_y  = Q_CROSS_H + 4 + LABEL_FONT_SIZE + 4
    mid_y    = Q_CROSS_H + 4 + (LABEL_FONT_SIZE + 4) // 2
    left_cx  = QCX // 2
    right_cx = QCX + (Q_CROSS_W - QCX) // 2
    _centred(draw, "TOP VIEW",   left_cx,  line1_y, lf)
    _centred(draw, "FRONT VIEW", left_cx,  line2_y, lf)
    _centred(draw, "END VIEW",   right_cx, mid_y,   lf)

    for letter, (x0, ccx, ccy) in zip("ABCD", choice_positions):
        _centred(draw, letter, ccx - VIEW // 2, mid_y, lf)

    canvas.save(str(out_path))


def make_composite_for_all(selected_questions_dir: Path) -> None:
    """Generate composite.png for every question subdirectory."""
    for q_dir in sorted(selected_questions_dir.iterdir()):
        if not q_dir.is_dir():
            continue
        out = q_dir / "composite.png"
        make_composite(q_dir, out)
        print(f"  composite: {out}")
