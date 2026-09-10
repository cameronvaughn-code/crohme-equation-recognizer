"""
preprocess.py

One-time offline step that turns the raw CROHME InkML tree into a flat,
fast-to-load training set:

  1. Collect .inkml files from explicit train / test directories.
  2. Parse each one (strokes + ground-truth LaTeX).
  3. De-duplicate by stroke-content hash (the CROHME releases overlap
     heavily year to year, and each test set ships with a duplicate "GT"
     copy).
  4. Render each expression to a PNG.
  5. Tokenize the LaTeX and build a vocabulary from the training split only.
  6. Write {train,val,test}.csv (image path, raw latex, tokens).

Why explicit directories instead of "glob everything":
`CROHME_full_v2` contains the 2011, 2012 and 2013 releases side by side.
The 2013 `TrainINKML` set is a superset of the earlier training data, and
the test sets each appear twice (`TestINKML` and `TestINKMLGT`). Globbing
the whole tree triples parts of the data and leaks test expressions into
train. The defaults below use the clean CROHME 2013 split:

  train:  CROHME2013_data/TrainINKML   (~8 800 expressions)
  test:   CROHME2013_data/TestINKML    (   671 expressions)

Usage:
  python preprocess.py --data_root ../data/extracted/CROHME_full_v2 \
      --out_dir ../data/processed --image_w 384 --image_h 128
"""

import argparse
import csv
import hashlib
import os
import random

from tqdm import tqdm

from inkml_parser import find_inkml_files, parse_inkml_file, tokenize_latex
from render import render_strokes
from vocab import Vocab

# Paths are relative to --data_root. TestINKML ships without labels; the
# ground-truth copies live in TestINKMLGT, so that's the test split.
DEFAULT_TRAIN_DIRS = ["CROHME2013_data/TrainINKML"]
DEFAULT_TEST_DIRS = ["CROHME2013_data/TestINKMLGT"]


def stroke_hash(strokes) -> str:
    """Content hash of an expression's strokes, for de-duplication."""
    h = hashlib.md5()
    for stroke in strokes:
        for x, y in stroke:
            h.update(f"{x:.1f},{y:.1f};".encode())
        h.update(b"|")
    return h.hexdigest()


def collect(dirs, data_root):
    files = []
    for rel in dirs:
        full = os.path.join(data_root, rel)
        if not os.path.isdir(full):
            raise SystemExit(f"Directory not found: {full}")
        found = find_inkml_files(full)
        print(f"  {rel}: {len(found)} .inkml files")
        files.extend(found)
    return files


def process_split(name, files, img_dir, image_w, image_h, seen_hashes):
    """Parse + render one split; returns (records, list_of_token_lists)."""
    records = []
    token_lists = []
    skipped = 0
    duplicates = 0

    for path in tqdm(files, desc=f"{name}: parse + render"):
        try:
            expr = parse_inkml_file(path)
        except Exception:
            skipped += 1
            continue

        if not expr.latex or not expr.strokes:
            skipped += 1
            continue

        tokens = tokenize_latex(expr.latex)
        if not tokens:
            skipped += 1
            continue

        h = stroke_hash(expr.strokes)
        if h in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(h)

        img = render_strokes(expr.strokes, out_w=image_w, out_h=image_h)
        base = os.path.splitext(os.path.basename(path))[0]
        img_name = f"{name}_{len(records):05d}_{base}.png"
        img.save(os.path.join(img_dir, img_name))

        records.append({
            "image": os.path.join("images", img_name),
            "latex": expr.latex,
            "tokens": " ".join(tokens),
        })
        token_lists.append(tokens)

    print(f"  {name}: kept {len(records)}, skipped {skipped}, "
          f"dropped {duplicates} duplicates")
    return records, token_lists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Path to extracted CROHME_full_v2 directory")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--image_w", type=int, default=384)
    parser.add_argument("--image_h", type=int, default=128)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--min_freq", type=int, default=2,
                        help="Min train frequency for a token to enter the vocab")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Cap total files (quick smoke tests)")
    parser.add_argument("--train_dirs", nargs="+", default=DEFAULT_TRAIN_DIRS)
    parser.add_argument("--test_dirs", nargs="+", default=DEFAULT_TEST_DIRS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Collecting training InkML:")
    train_files = collect(args.train_dirs, args.data_root)
    print("Collecting test InkML:")
    test_files = collect(args.test_dirs, args.data_root)

    random.shuffle(train_files)
    random.shuffle(test_files)
    if args.max_files:
        keep_test = max(1, args.max_files // 10)
        test_files = test_files[:keep_test]
        train_files = train_files[: args.max_files - len(test_files)]
        print(f"Capped to {len(train_files)} train / {len(test_files)} test files")

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # De-dup within and across splits; process test first so a leaked
    # expression is kept on the test side, never in train.
    seen_hashes: set = set()
    test_records, _ = process_split("test", test_files, img_dir,
                                    args.image_w, args.image_h, seen_hashes)
    train_all, train_tokens_all = process_split("train", train_files, img_dir,
                                                args.image_w, args.image_h, seen_hashes)

    # Carve a validation set out of the (already shuffled) training records.
    n_val = int(len(train_all) * args.val_fraction)
    val_records = train_all[:n_val]
    train_records = train_all[n_val:]
    train_tokens = train_tokens_all[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    for split_name, recs in [("train", train_records),
                             ("val", val_records),
                             ("test", test_records)]:
        csv_path = os.path.join(args.out_dir, f"{split_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "latex", "tokens"])
            writer.writeheader()
            writer.writerows(recs)
        print(f"Wrote {len(recs):>6} rows -> {csv_path}")

    vocab = Vocab.build(train_tokens, min_freq=args.min_freq)
    vocab_path = os.path.join(args.out_dir, "vocab.json")
    vocab.save(vocab_path)
    print(f"Vocab size: {len(vocab)} (min_freq={args.min_freq}) -> {vocab_path}")


if __name__ == "__main__":
    main()
