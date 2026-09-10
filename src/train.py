"""
train.py

Trains the Image2LatexModel on the preprocessed CROHME data.

Usage:
  python train.py --data_dir ../data/processed --epochs 40 --batch_size 32

Runs on CUDA, Apple-Silicon MPS, or CPU automatically. Every epoch it
writes:
  checkpoints/last.pt          - latest weights + config + optimizer state
  checkpoints/best.pt          - best val token-accuracy so far
  checkpoints/history.json     - per-epoch metrics, for plot_history.py
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CrohmeDataset, make_collate_fn
from device import get_device
from metrics import MetricAccumulator
from model import Image2LatexModel
from vocab import Vocab, PAD, SOS, EOS


@torch.no_grad()
def evaluate(model, loader, vocab, device, beam_size=1, max_batches=None):
    model.eval()
    acc = MetricAccumulator()
    sos_id, eos_id = vocab.token_to_id[SOS], vocab.token_to_id[EOS]
    for i, (images, targets, _lengths) in enumerate(loader):
        images = images.to(device)
        preds = model.predict(images, sos_id, eos_id, beam_size=beam_size)
        for b in range(images.size(0)):
            pred_tokens = vocab.decode(preds[b])
            true_tokens = vocab.decode(targets[b].tolist())
            acc.update(pred_tokens, true_tokens)
        if max_batches and i + 1 >= max_batches:
            break
    model.train()
    return acc.as_dict()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone_lr_mult", type=float, default=0.25,
                   help="LR multiplier for the (unfrozen) pretrained ResNet layers")
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--image_w", type=int, default=384)
    p.add_argument("--image_h", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--encoder_channels", type=int, default=256)
    p.add_argument("--decoder", choices=["transformer", "gru"], default="transformer")
    p.add_argument("--dec_layers", type=int, default=4)
    p.add_argument("--dec_heads", type=int, default=8)
    p.add_argument("--dec_ff", type=int, default=1024)
    p.add_argument("--dec_dropout", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--tf_start", type=float, default=1.0, help="Teacher-forcing ratio during the hold phase (GRU decoder only)")
    p.add_argument("--tf_end", type=float, default=0.6, help="Teacher-forcing ratio after decay")
    p.add_argument("--tf_hold_epochs", type=int, default=12,
                   help="Epochs at tf_start before scheduled sampling begins")
    p.add_argument("--tf_decay_epochs", type=int, default=18)
    p.add_argument("--augment", action="store_true", help="Train-time image augmentation")
    p.add_argument("--val_batches", type=int, default=None, help="Cap val batches per epoch (None = full val)")
    p.add_argument("--checkpoint_dir", default="../checkpoints")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default=None, help="Force cuda / mps / cpu")
    p.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    args = p.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    vocab = Vocab.load(os.path.join(args.data_dir, "vocab.json"))
    pad_id = vocab.token_to_id[PAD]
    print(f"Vocab size: {len(vocab)}")

    train_ds = CrohmeDataset(os.path.join(args.data_dir, "train.csv"), args.data_dir,
                             vocab, args.image_w, args.image_h, augment=args.augment)
    val_ds = CrohmeDataset(os.path.join(args.data_dir, "val.csv"), args.data_dir,
                           vocab, args.image_w, args.image_h, augment=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  (augment={args.augment})")

    collate_fn = make_collate_fn(pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    model = Image2LatexModel(
        vocab_size=len(vocab), pad_id=pad_id,
        encoder_channels=args.encoder_channels,
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        decoder=args.decoder, dec_layers=args.dec_layers, dec_heads=args.dec_heads,
        dec_ff=args.dec_ff, dec_dropout=args.dec_dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.decoder} decoder, {n_params/1e6:.2f}M params")

    # Two param groups: the pretrained ResNet layers fine-tune at a lower
    # LR than the from-scratch attention decoder.
    backbone_ids = {id(p) for p in model.encoder.parameters()}
    backbone_params = [p for p in model.parameters()
                       if id(p) in backbone_ids and p.requires_grad]
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr},
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
    ], weight_decay=args.weight_decay)

    # Linear warmup then cosine decay to ~3% of base LR. Deterministic, so
    # a noisy validation metric can't stall training (an earlier plateau
    # scheduler froze the LR at ~2e-5 by epoch 20 and the model undertrained).
    def lr_factor(epoch):  # epoch is 0-indexed here
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(args.warmup_epochs, 1)
        prog = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id,
                                    label_smoothing=args.label_smoothing)

    def tf_ratio(epoch):  # 1-indexed; the transformer decoder ignores this
        if args.decoder != "gru":
            return 1.0
        t = min(max(epoch - 1 - args.tf_hold_epochs, 0) / max(args.tf_decay_epochs, 1), 1.0)
        return args.tf_start + t * (args.tf_end - args.tf_start)

    config = {k: getattr(args, k) for k in
              ["image_w", "image_h", "hidden_dim", "embed_dim", "encoder_channels",
               "decoder", "dec_layers", "dec_heads", "dec_ff"]}
    config["vocab_size"] = len(vocab)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history = []
    start_epoch = 1
    best_token_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_token_acc = ckpt.get("best_token_acc", 0.0)
        for _ in range(start_epoch - 1):
            scheduler.step()
        hist_path = os.path.join(args.checkpoint_dir, "history.json")
        if os.path.exists(hist_path):
            history = json.load(open(hist_path))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        tf = tf_ratio(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, targets, _lengths in pbar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(images, targets, teacher_forcing_ratio=tf)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", tf=f"{tf:.2f}")

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, vocab, device,
                               beam_size=1, max_batches=args.val_batches)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        dt = time.time() - t0
        print(f"Epoch {epoch}: loss={avg_loss:.4f}  "
              f"val_exact={val_metrics['exact_match']:.4f}  "
              f"val_token_acc={val_metrics['token_accuracy']:.4f}  "
              f"lr={lr_now:.2e}  tf={tf:.2f}  ({dt:.0f}s)")

        row = {"epoch": epoch, "train_loss": round(avg_loss, 4), "tf": round(tf, 3),
               "lr": lr_now, "seconds": round(dt, 1), **val_metrics}
        history.append(row)
        json.dump(history, open(os.path.join(args.checkpoint_dir, "history.json"), "w"), indent=2)

        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "epoch": epoch, "config": config, "val_metrics": val_metrics,
                   "best_token_acc": best_token_acc}
        torch.save(payload, os.path.join(args.checkpoint_dir, "last.pt"))

        if val_metrics["token_accuracy"] > best_token_acc:
            best_token_acc = val_metrics["token_accuracy"]
            payload["best_token_acc"] = best_token_acc
            torch.save(payload, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  New best (val_token_acc={best_token_acc:.4f}) saved")

    print(f"Done. Best val token accuracy: {best_token_acc:.4f}")


if __name__ == "__main__":
    main()
