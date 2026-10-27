import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from keypoint_sequences import FEATURES_PER_FRAME, KeypointSequenceDataset, read_keypoint_manifest
from metrics import classification_metrics, predict_logits
from classifier import SignClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3: train a word-level sign classifier on keypoint sequences."
    )
    parser.add_argument("--keypoints", required=True, help="manifest_keypoints.csv from predict_wlasl.py or refine_keypoints.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--arch", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = {split: read_keypoint_manifest(args.keypoints, split) for split in ("train", "val", "test")}
    for split, rows in splits.items():
        print(f"{split} clips: {len(rows)}")

    train_loader = DataLoader(KeypointSequenceDataset(splits["train"], train=True), batch_size=args.batch_size, shuffle=True)
    eval_loaders = {
        split: DataLoader(KeypointSequenceDataset(splits[split]), batch_size=args.batch_size)
        for split in ("val", "test")
    }

    model = SignClassifier(
        FEATURES_PER_FRAME, args.num_classes, args.arch, args.hidden, args.num_layers, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    log_rows = []
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        losses = []
        for features, labels in train_loader:
            loss = loss_fn(model(features.to(device)), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        val = classification_metrics(*predict_logits(model, eval_loaders["val"], device), args.num_classes)
        log_rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}})

        if val["top1"] > best_val:
            best_val = val["top1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, os.path.join(args.out_dir, "best.pt"))

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch}: train_loss {log_rows[-1]['train_loss']:.4f} "
                f"val top1 {val['top1']:.3f} top5 {val['top5']:.3f} ({time.time() - start:.1f}s)"
            )

    with open(os.path.join(args.out_dir, "train_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    checkpoint = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    results = {"best_epoch": checkpoint["epoch"]}
    for split, loader in eval_loaders.items():
        results[split] = classification_metrics(*predict_logits(model, loader, device), args.num_classes)

    with open(os.path.join(args.out_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Best val epoch: {results['best_epoch']}")
    for split in ("val", "test"):
        m = results[split]
        print(f"{split}: top1 {m['top1']:.3f} top5 {m['top5']:.3f} macro_f1 {m['macro_f1']:.3f} (n={m['n']})")


if __name__ == "__main__":
    main()
