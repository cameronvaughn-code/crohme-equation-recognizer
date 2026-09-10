"""
evaluate.py

Full evaluation on a processed split (default: test). Reports exact-match
and token accuracy, and writes a per-example predictions CSV plus a
summary of the most common substitution errors.

Usage:
  python evaluate.py --checkpoint ../checkpoints/best.pt \
      --data_dir ../data/processed --split test --beam_size 5
"""

import argparse
import csv
import os
from collections import Counter

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CrohmeDataset, make_collate_fn
from device import get_device
from metrics import MetricAccumulator, levenshtein
from predict import build_model_from_checkpoint
from vocab import Vocab, PAD, SOS, EOS


def alignment_ops(pred, ref):
    """Backtrace a Levenshtein alignment; yield ('sub'|'del'|'ins', a, b)."""
    n, m = len(pred), len(ref)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == ref[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (pred[i - 1] != ref[j - 1]):
            if pred[i - 1] != ref[j - 1]:
                yield ("sub", pred[i - 1], ref[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            yield ("del", pred[i - 1], None)
            i -= 1
        else:
            yield ("ins", None, ref[j - 1])
            j -= 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_dir", default="../outputs")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    vocab = Vocab.load(os.path.join(args.data_dir, "vocab.json"))
    sos_id, eos_id = vocab.token_to_id[SOS], vocab.token_to_id[EOS]

    ckpt = torch.load(args.checkpoint, map_location=device)
    model, (image_w, image_h) = build_model_from_checkpoint(ckpt, vocab, device)

    ds = CrohmeDataset(os.path.join(args.data_dir, f"{args.split}.csv"),
                       args.data_dir, vocab, image_w, image_h, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=make_collate_fn(vocab.token_to_id[PAD]))
    print(f"Evaluating {len(ds)} {args.split} examples on {device} "
          f"(beam_size={args.beam_size})")

    acc = MetricAccumulator()
    error_counter = Counter()
    rows = []
    for images, targets, _lengths in tqdm(loader, desc="eval"):
        images = images.to(device)
        preds = model.predict(images, sos_id, eos_id, beam_size=args.beam_size)
        for b in range(images.size(0)):
            pred_tokens = vocab.decode(preds[b])
            ref_tokens = vocab.decode(targets[b].tolist())
            acc.update(pred_tokens, ref_tokens)
            for op, a, c in alignment_ops(pred_tokens, ref_tokens):
                error_counter[(op, a, c)] += 1
            rows.append({
                "image": ds.rows[len(rows)]["image"],
                "reference": " ".join(ref_tokens),
                "prediction": " ".join(pred_tokens),
                "exact": int(pred_tokens == ref_tokens),
                "edit_distance": levenshtein(pred_tokens, ref_tokens),
                "ref_len": len(ref_tokens),
            })

    os.makedirs(args.out_dir, exist_ok=True)
    pred_path = os.path.join(args.out_dir, f"predictions_{args.split}.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "reference", "prediction",
                                          "exact", "edit_distance", "ref_len"])
        w.writeheader()
        w.writerows(rows)

    summary = acc.as_dict()
    print("\n=== Results ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nPer-example predictions -> {pred_path}")

    print("\nTop 15 error patterns (op, predicted, reference):")
    for (op, a, c), n in error_counter.most_common(15):
        print(f"  {n:>4}  {op:<4} {str(a):<10} -> {str(c)}")


if __name__ == "__main__":
    main()
