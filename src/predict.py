"""
predict.py

Run inference on a single image (a rendered CROHME PNG, or a photo/scan of
your own handwritten equation) and print the predicted LaTeX.

Usage:
  python predict.py --checkpoint ../checkpoints/best.pt \
      --vocab ../data/processed/vocab.json \
      --image path/to/equation.png --beam_size 5
"""

import argparse

import numpy as np
import torch
from PIL import Image, ImageOps

from device import get_device
from model import Image2LatexModel
from vocab import Vocab, PAD, SOS, EOS


def load_image(path: str, image_w: int, image_h: int,
               invert_if_needed: bool = True) -> torch.Tensor:
    img = Image.open(path).convert("L")
    # CROHME renders are dark ink on white. If someone passes a photo that's
    # light ink on dark, flip it so it matches training data.
    arr = np.asarray(img, dtype=np.float32)
    if invert_if_needed and arr.mean() < 110:
        img = ImageOps.invert(img)
    img = img.resize((image_w, image_h))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def build_model_from_checkpoint(ckpt, vocab, device):
    cfg = ckpt.get("config", {})
    model = Image2LatexModel(
        vocab_size=len(vocab),
        pad_id=vocab.token_to_id[PAD],
        encoder_channels=cfg.get("encoder_channels", 256),
        embed_dim=cfg.get("embed_dim", 128),
        hidden_dim=cfg.get("hidden_dim", 256),
        decoder=cfg.get("decoder", "gru"),
        dec_layers=cfg.get("dec_layers", 4),
        dec_heads=cfg.get("dec_heads", 8),
        dec_ff=cfg.get("dec_ff", 1024),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, (cfg.get("image_w", 384), cfg.get("image_h", 128))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    vocab = Vocab.load(args.vocab)
    sos_id, eos_id = vocab.token_to_id[SOS], vocab.token_to_id[EOS]

    ckpt = torch.load(args.checkpoint, map_location=device)
    model, (image_w, image_h) = build_model_from_checkpoint(ckpt, vocab, device)

    image_tensor = load_image(args.image, image_w, image_h).to(device)
    preds = model.predict(image_tensor, sos_id, eos_id, beam_size=args.beam_size)
    latex = " ".join(vocab.decode(preds[0]))
    print(f"Predicted LaTeX: {latex}")


if __name__ == "__main__":
    main()
