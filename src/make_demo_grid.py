"""
make_demo_grid.py

Build a portfolio-friendly montage of test-set predictions: each cell is
the input image with its reference LaTeX and the model's prediction,
colour-coded green (exact match) or red (mismatch).

Reads outputs/predictions_test.csv (from evaluate.py) plus the processed
images. Picks a mix of correct and incorrect examples.

Usage:
  python make_demo_grid.py --data_dir ../data/processed \
      --predictions ../outputs/predictions_test.csv --out ../outputs/demo_grid.png
"""

import argparse
import csv
import os
import random
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--predictions", default="../outputs/predictions_test.csv")
    p.add_argument("--out", default="../outputs/demo_grid.png")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    with open(args.predictions, newline="") as f:
        rows = list(csv.DictReader(f))

    correct = [r for r in rows if r["exact"] == "1"]
    wrong = [r for r in rows if r["exact"] == "0"]
    random.shuffle(correct)
    random.shuffle(wrong)

    n = args.rows * args.cols
    n_correct = min(len(correct), (n + 1) // 2)
    pick = correct[:n_correct] + wrong[: n - n_correct]
    random.shuffle(pick)

    fig, axes = plt.subplots(args.rows, args.cols,
                             figsize=(3.4 * args.cols, 3.0 * args.rows))
    axes = axes.reshape(-1)
    for ax, r in zip(axes, pick):
        img = Image.open(os.path.join(args.data_dir, r["image"])).convert("L")
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ok = r["exact"] == "1"
        colour = "#1a7f37" if ok else "#cf222e"
        wrap = lambda s: "\n      ".join(textwrap.wrap(s, 46)) or ""
        title = f"ref:  {wrap(r['reference'])}\npred: {wrap(r['prediction'])}"
        ax.set_title(title, fontsize=7, color=colour, loc="left", family="monospace")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(colour)
            s.set_linewidth(2.5)
    for ax in axes[len(pick):]:
        ax.axis("off")

    fig.suptitle("CROHME 2013 test set — sample predictions "
                 "(green = exact match, red = mismatch)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
