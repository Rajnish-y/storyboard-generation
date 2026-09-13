"""
storyboard_composer.py
------------------------------------------------------------------------
Stage 7: Storyboard Composition. Takes the individual scene PNGs from
storyboard_output/ and composites them into a single storyboard sheet —
a grid layout with scene numbers and (optionally) captions underneath
each frame, plus a PDF export.

Run this locally (no GPU needed) or in Colab, after storyboard_generator.py
has produced storyboard_output/.

Setup:
    pip install pillow

Usage:
    python storyboard_composer.py --scenes_json scenes.json --frames_dir storyboard_output
"""

import argparse
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int):
    # Falls back to PIL's default bitmap font if no TTF is found on the system
    for candidate in ["DejaVuSans-Bold.ttf", "Arial.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def compose_storyboard(
    scenes_json_path: str,
    frames_dir: str,
    out_path: str = "storyboard_sheet.png",
    columns: int = 2,
    thumb_width: int = 512,
    caption_height: int = 90,
    padding: int = 20,
    background_color=(20, 20, 20),
    text_color=(255, 255, 255),
):
    with open(scenes_json_path) as f:
        data = json.load(f)
    scenes = data["scenes"]

    frames = []
    for scene in scenes:
        path = os.path.join(frames_dir, f"{scene['scene_id']}.png")
        if not os.path.exists(path):
            print(f"Warning: missing frame for {scene['scene_id']} at {path}, skipping.")
            continue
        img = Image.open(path).convert("RGB")
        aspect = img.height / img.width
        thumb_height = int(thumb_width * aspect)
        img = img.resize((thumb_width, thumb_height))
        frames.append((scene, img))

    if not frames:
        raise SystemExit("No frames found — check frames_dir and scenes_json paths.")

    rows = math.ceil(len(frames) / columns)
    cell_w = thumb_width + padding
    cell_h = max(f[1].height for f in frames) + caption_height + padding

    sheet_w = cell_w * columns + padding
    sheet_h = cell_h * rows + padding

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=background_color)
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(22)
    caption_font = load_font(16)

    for idx, (scene, img) in enumerate(frames):
        col = idx % columns
        row = idx // columns
        x = padding + col * cell_w
        y = padding + row * cell_h

        sheet.paste(img, (x, y))

        label = f"{idx + 1}. {scene['scene_id']}"
        draw.text((x, y + img.height + 8), label, font=title_font, fill=text_color)

        caption = scene.get("prompt", "")
        if len(caption) > 90:
            caption = caption[:87] + "..."
        draw.text((x, y + img.height + 36), caption, font=caption_font, fill=(200, 200, 200))

    sheet.save(out_path)
    print(f"Saved storyboard sheet -> {out_path}")

    pdf_path = os.path.splitext(out_path)[0] + ".pdf"
    sheet.save(pdf_path, "PDF")
    print(f"Saved PDF -> {pdf_path}")

    return out_path, pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7: compose generated frames into a storyboard sheet")
    parser.add_argument("--scenes_json", type=str, default="scenes.json")
    parser.add_argument("--frames_dir", type=str, default="storyboard_output")
    parser.add_argument("--out", type=str, default="storyboard_sheet.png")
    parser.add_argument("--columns", type=int, default=2)
    args = parser.parse_args()

    compose_storyboard(
        scenes_json_path=args.scenes_json,
        frames_dir=args.frames_dir,
        out_path=args.out,
        columns=args.columns,
    )