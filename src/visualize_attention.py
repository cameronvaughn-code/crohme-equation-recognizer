"""
visualize_attention.py

Decode one image and render the decoder's attention over the equation as
it produces each LaTeX token. Each panel is the input image with that
step's attention map (upsampled from the 16x16 encoder grid) overlaid.

Usage:
  python visualize_attention.py --checkpoint ../checkpoints/best.pt \
      --vocab ../data/processed/vocab.json \
      --image ../data/processed/images/test_00000_*.png \
      --out ../outputs/attention.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from device import get_device
from predict import build_model_from_checkpoint, load_image
from vocab import Vocab, SOS, EOS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="../outputs/attention.png")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    vocab = Vocab.load(args.vocab)
    sos_id, eos_id = vocab.token_to_id[SOS], vocab.token_to_id[EOS]

    ckpt = torch.load(args.checkpoint, map_location=device)
    model, (image_w, image_h) = build_model_from_checkpoint(ckpt, vocab, device)
    gh, gw = model.encode_grid_shape(image_w, image_h)

    img_tensor = load_image(args.image, image_w, image_h).to(device)
    seqs, attns = model.predict_with_attention(img_tensor, sos_id, eos_id)
    token_ids = seqs[0]
    maps = attns[0]

    tokens = [vocab.id_to_token.get(i, "<unk>") for i in token_ids]
    # Drop the trailing <eos> panel.
    if tokens and token_ids[-1] == eos_id:
        tokens, maps = tokens[:-1], maps[:-1]

    base_img = img_tensor.squeeze().cpu().numpy()
    # Crop away the vertical white margins so each panel shows the ink big.
    ink_rows = np.where((base_img < 0.5).any(axis=1))[0]
    if len(ink_rows):
        pad = 8
        r0, r1 = max(int(ink_rows[0]) - pad, 0), min(int(ink_rows[-1]) + pad, image_h)
    else:
        r0, r1 = 0, image_h
    base_crop = base_img[r0:r1]
    ch = r1 - r0

    n = max(len(tokens), 1)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(3.0 * cols, 3.2 * cols * ch / image_w + 0.5))
    axes = np.array(axes).reshape(-1)

    for k in range(len(axes)):
        ax = axes[k]
        ax.axis("off")
        if k >= len(tokens):
            continue
        amap = maps[k].numpy().reshape(gh, gw)
        amap = amap / (amap.max() + 1e-8)
        amap = np.kron(amap, np.ones((image_h // gh, image_w // gw)))[r0:r1]
        ax.imshow(base_crop, cmap="gray")
        # Alpha scales with attention weight, so low-attention areas stay
        # readable and the hotspot pops.
        ax.imshow(amap, cmap="jet", alpha=0.6 * amap,
                  extent=(0, image_w, ch, 0))
        ax.set_title(tokens[k], fontsize=11)

    pred = " ".join(t for t in tokens)
    fig.suptitle(f"Prediction:  {pred}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Wrote {args.out}")
    print(f"Predicted LaTeX: {pred}")


if __name__ == "__main__":
    main()
