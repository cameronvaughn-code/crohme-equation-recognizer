"""
plot_history.py

Plot training curves from checkpoints/history.json (written by train.py).

Usage:
  python plot_history.py --history ../checkpoints/history.json --out ../outputs/training_curves.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", default="../checkpoints/history.json")
    p.add_argument("--out", default="../outputs/training_curves.png")
    args = p.parse_args()

    hist = json.load(open(args.history))
    epochs = [r["epoch"] for r in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [r["train_loss"] for r in hist], marker="o")
    ax1.set_title("Training loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [r["token_accuracy"] for r in hist], marker="o", label="token accuracy")
    ax2.plot(epochs, [r["exact_match"] for r in hist], marker="s", label="exact match")
    ax2.set_title("Validation metrics")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("fraction")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
