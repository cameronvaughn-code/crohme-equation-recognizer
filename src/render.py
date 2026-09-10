"""
render.py

Renders a list of strokes (from inkml_parser) into a grayscale bitmap image,
the way the CROHME "offline" data is produced from the "online" stroke data.

This is what turns the raw pen-coordinate sequences into something a CNN
encoder can consume.
"""

from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

Point = Tuple[float, float]
Stroke = List[Point]


def render_strokes(
    strokes: List[Stroke],
    out_w: int = 384,
    out_h: int = 128,
    padding: int = 8,
    line_width: int = 3,
) -> Image.Image:
    """
    Render strokes onto a white-background grayscale image with black ink,
    preserving aspect ratio and centering the expression.

    Math expressions are wide (median aspect ratio ~3:1 in CROHME), so the
    default canvas is a 3:1 landscape rectangle rather than a square. A
    square canvas squashes long expressions into a few pixels of height,
    which is the single biggest thing that hurt an earlier version of this
    model.
    """
    if not strokes or all(len(s) == 0 for s in strokes):
        return Image.new("L", (out_w, out_h), color=255)

    all_points = [p for stroke in strokes for p in stroke]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)

    # Largest scale that fits the ink inside the padded canvas, aspect preserved.
    draw_w = out_w - 2 * padding
    draw_h = out_h - 2 * padding
    scale = min(draw_w / width, draw_h / height)

    offset_x = padding + (draw_w - width * scale) / 2
    offset_y = padding + (draw_h - height * scale) / 2

    img = Image.new("L", (out_w, out_h), color=255)
    draw = ImageDraw.Draw(img)

    for stroke in strokes:
        if len(stroke) == 0:
            continue
        transformed = [
            (
                (x - min_x) * scale + offset_x,
                (y - min_y) * scale + offset_y,
            )
            for x, y in stroke
        ]
        if len(transformed) == 1:
            x, y = transformed[0]
            r = line_width / 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
        else:
            draw.line(transformed, fill=0, width=line_width, joint="curve")

    return img


def image_to_array(img: Image.Image) -> np.ndarray:
    """Convert to a normalized float32 array in [0, 1], shape (H, W)."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


if __name__ == "__main__":
    import sys
    from inkml_parser import parse_inkml_file

    if len(sys.argv) != 3:
        print("Usage: python render.py <path_to_inkml> <output_png_path>")
        sys.exit(1)

    expr = parse_inkml_file(sys.argv[1])
    img = render_strokes(expr.strokes, out_w=768, out_h=256)
    img.save(sys.argv[2])
    print(f"Rendered '{expr.latex}' -> {sys.argv[2]}")
