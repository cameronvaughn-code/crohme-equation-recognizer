"""
dataset.py

PyTorch Dataset + collate_fn for the preprocessed CROHME data (see
preprocess.py). Each item is (image_tensor, token_id_tensor); batches are
padded to the longest sequence in the batch.

Augmentation (train split only) stands in for writer variation, which is
what the model most needs to be robust to: affine + perspective warp,
stroke erosion/dilation, and speckle noise. See CrohmeDataset._augment.
"""

import csv
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vocab import Vocab


def _perspective_coeffs(src, dst):
    """Solve the 8 PIL PERSPECTIVE coefficients mapping src -> dst quads."""
    import numpy as _np
    A = []
    for (xs, ys), (xd, yd) in zip(src, dst):
        A.append([xs, ys, 1, 0, 0, 0, -xd * xs, -xd * ys])
        A.append([0, 0, 0, xs, ys, 1, -yd * xs, -yd * ys])
    A = _np.array(A, dtype=_np.float64)
    b = _np.array([c for pt in dst for c in pt], dtype=_np.float64)
    return _np.linalg.solve(A, b).tolist()


class CrohmeDataset(Dataset):
    def __init__(self, csv_path: str, images_root: str, vocab: Vocab,
                 image_w: int = 384, image_h: int = 128, augment: bool = False):
        self.rows: List[dict] = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        self.images_root = images_root
        self.vocab = vocab
        self.image_w = image_w
        self.image_h = image_h
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, rel_path: str) -> Image.Image:
        img = Image.open(os.path.join(self.images_root, rel_path)).convert("L")
        if img.size != (self.image_w, self.image_h):
            img = img.resize((self.image_w, self.image_h))
        return img

    def _augment(self, img: Image.Image) -> Image.Image:
        """
        Aggressive augmentation to stand in for writer variation, which is
        the thing the model most needs to be robust to (CROHME's test
        writers are disjoint from its training writers).
        """
        w, h = img.size

        # Affine: rotate +/-8 deg, anisotropic scale 0.8-1.2, translate +/-8%.
        angle = random.uniform(-8, 8)
        sx = random.uniform(0.8, 1.2)
        sy = random.uniform(0.85, 1.2)
        tx = random.uniform(-0.08, 0.08) * w
        ty = random.uniform(-0.08, 0.08) * h
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
        img = img.transform((w, h), Image.AFFINE,
                            (1 / sx, 0, -tx, 0, 1 / sy, -ty),
                            resample=Image.BILINEAR, fillcolor=255)

        # Random perspective warp (mild).
        if random.random() < 0.5:
            m = 0.06
            dx = [random.uniform(-m, m) * w for _ in range(4)]
            dy = [random.uniform(-m, m) * h for _ in range(4)]
            src = [(0, 0), (w, 0), (w, h), (0, h)]
            dst = [(src[i][0] + dx[i], src[i][1] + dy[i]) for i in range(4)]
            coeffs = _perspective_coeffs(dst, src)
            img = img.transform((w, h), Image.PERSPECTIVE, coeffs,
                                resample=Image.BILINEAR, fillcolor=255)

        arr = np.asarray(img).astype(np.uint8)
        ink = arr < 128

        # Stroke-width jitter: thicken (dilate) or thin (erode).
        r = random.random()
        if r < 0.35:
            d = ink.copy()
            d[1:, :] |= ink[:-1, :]; d[:-1, :] |= ink[1:, :]
            d[:, 1:] |= ink[:, :-1]; d[:, :-1] |= ink[:, 1:]
            ink = d
        elif r < 0.5:
            e = ink & np.roll(ink, 1, 0) & np.roll(ink, -1, 0) \
                    & np.roll(ink, 1, 1) & np.roll(ink, -1, 1)
            ink = e
        arr = np.where(ink, 0, 255).astype(np.uint8)

        # Speckle noise.
        if random.random() < 0.3:
            noise = np.random.rand(h, w) < 0.01
            arr[noise] = 0
        return Image.fromarray(arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        img = self._load_image(row["image"])
        if self.augment:
            img = self._augment(img)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

        tokens = row["tokens"].split() if row["tokens"] else []
        token_ids = self.vocab.encode(tokens)  # adds <sos> / <eos>
        return img_tensor, torch.tensor(token_ids, dtype=torch.long)


class Collate:
    """Pads a batch to its longest sequence. A class (not a closure) so it
    survives pickling for DataLoader workers under macOS 'spawn'."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        images, seqs = zip(*batch)
        images = torch.stack(images, dim=0)  # (B, 1, H, W)
        max_len = max(s.size(0) for s in seqs)
        padded = torch.full((len(seqs), max_len), self.pad_id, dtype=torch.long)
        lengths = torch.zeros(len(seqs), dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, : s.size(0)] = s
            lengths[i] = s.size(0)
        return images, padded, lengths


def make_collate_fn(pad_id: int) -> Collate:
    return Collate(pad_id)
