#!/usr/bin/env python3
"""Create Week 11 paper-preview figures from existing Week 8 PNGs.

This script does not require Knox checkpoints. It wraps the already generated
Week 8 figures with consistent TrueType typography, explicit guardrail notes,
and 300 dpi output. The originals remain untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\calibri.ttf"),
]
BOLD_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path(r"C:\Windows\Fonts\segoeuib.ttf"),
    Path(r"C:\Windows\Fonts\calibrib.ttf"),
]


FIGURES = [
    {
        "src": "hero_ann_arbor_seed42.png",
        "dst": "hero_ann_arbor_polished_seed42.png",
        "title": "Ann Arbor qualitative example",
        "subtitle": "This scene shows +0.93 dB; the three-seed mean gain is +0.571 +/- 0.157 dB.",
    },
    {
        "src": "failure_cases_ann_arbor_seed42.png",
        "dst": "failure_cases_ann_arbor_polished_seed42.png",
        "title": "Failure cases selected by paired PSNR delta",
        "subtitle": "Rows are cases where the method underperforms the no-registration baseline most.",
    },
    {
        "src": "cross_dataset_gallery_seed42.png",
        "dst": "cross_dataset_gallery_polished_seed42.png",
        "title": "Cross-dataset qualitative context",
        "subtitle": "Kust4K row is completeness only: +0.096 +/- 0.067 dB within-dataset gain, not significant.",
    },
]


def load_font(candidates: list[Path], size: int) -> ImageFont.ImageFont:
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def polish_one(src: Path, dst: Path, title: str, subtitle: str, title_font, body_font) -> None:
    image = Image.open(src).convert("RGB")
    margin_x = 34
    top_h = 102
    bottom_h = 34
    canvas = Image.new("RGB", (image.width, image.height + top_h + bottom_h), "white")
    canvas.paste(image, (0, top_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.width, top_h), fill=(248, 250, 252))
    draw.text((margin_x, 18), title, fill=(17, 24, 39), font=title_font)
    for idx, line in enumerate(wrap_text(draw, subtitle, body_font, image.width - 2 * margin_x)):
        draw.text((margin_x, 58 + idx * 24), line, fill=(55, 65, 81), font=body_font)
    draw.rectangle((0, canvas.height - bottom_h, image.width, canvas.height), fill=(248, 250, 252))
    draw.text((margin_x, canvas.height - 26), "Paper preview; quantitative claims come from Tables 2-4.", fill=(75, 85, 99), font=body_font)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, dpi=(300, 300))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="figures/week8")
    parser.add_argument("--output-dir", default="figures/week8")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    title_font = load_font(BOLD_CANDIDATES, 26)
    body_font = load_font(FONT_CANDIDATES, 18)

    for spec in FIGURES:
        src = in_dir / spec["src"]
        dst = out_dir / spec["dst"]
        if not src.exists():
            raise FileNotFoundError(src)
        polish_one(src, dst, spec["title"], spec["subtitle"], title_font, body_font)
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
