import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "recognition"))

from frame_dataset import FrameClipDataset, read_frame_manifest  # noqa: E402
from frame_model import FrameGRUClassifier  # noqa: E402
from metrics import classification_metrics, predict_logits  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raw-frame baseline: ImageNet ResNet-18 on each frame + GRU, trained on clean frames."
    )
    parser.add_argument("--manifest", required=True, help="manifest_frames.csv from sample_frames.py (16 frames per clip)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the GRU head")
    parser.add_argument("--backbone_lr", type=float, default=1e-4, help="Learning rate for the ResNet-18 backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true", help="Train only the GRU head on frozen ImageNet features")
    parser.add_argument("--no_pretrained", action="store_true", help="Random-init the backbone (offline smoke tests)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = {split: read_frame_manifest(args.manifest, split) for split in ("train", "val", "test")}
    for split, rows in splits.items():
        print(f"{split} clips: {len(rows)}")

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(FrameClipDataset(splits["train"], train=True), shuffle=True, **loader_args)
    eval_loaders = {split: DataLoader(FrameClipDataset(splits[split]), **loader_args) for split in ("val", "test")}

    model = FrameGRUClassifier(
        args.num_classes, args.hidden, args.num_layers, args.dropout,
        pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone,
    ).to(device)
    groups = [{"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": args.lr}]
    if not args.freeze_backbone:
        groups.append({"params": model.backbone.parameters(), "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    log_rows = []
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        losses = []
        for frames, labels in train_loader:
            loss = loss_fn(model(frames.to(device)), labels.to(device))
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
