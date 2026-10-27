import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from coco_wholebody_dataset import CocoWholeBodyKeypoints, load_samples, split_by_image_id
from keypoint_spec import GROUP_SLICES, NUM_KEYPOINTS
from model import KeypointHeatmapNet, weighted_mse
from transforms import apply_affine, decode_heatmaps, get_affine

PCK_THRESHOLDS = (0.05, 0.1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Stage 1 spatial keypoint extractor on COCO-WholeBody."
    )
    parser.add_argument("--train_ann", required=True)
    parser.add_argument("--train_img_dir", required=True)
    parser.add_argument("--val_ann", help="Defaults to a held-out image_id split of --train_ann.")
    parser.add_argument("--val_img_dir")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=192)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0, help="0 = full epoch")
    parser.add_argument("--max_eval_batches", type=int, default=0, help="0 = full val set")
    parser.add_argument("--resume", help="Path to a last.pt checkpoint to continue from.")
    parser.add_argument("--no_amp", action="store_true", help="Disable bf16 autocast on CUDA")
    return parser.parse_args()


def autocast(device, enabled):
    # bf16 needs no loss scaling; the loss itself is computed in float32 below.
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


def predict_original_coords(heatmaps, centers, sizes, input_w, input_h, stride):
    coords, scores = decode_heatmaps(heatmaps, stride)
    out = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        inverse = get_affine(centers[i], sizes[i], 0.0, input_w, input_h, inverse=True)
        out[i] = apply_affine(coords[i], inverse)
    return out, scores


def evaluate(model, loader, device, args, stride, max_batches=0):
    model.eval()
    norm_errors = []
    labeled = []
    losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            with autocast(device, not args.no_amp):
                pred = model(images)
            pred = pred.float()
            losses.append(
                weighted_mse(pred, batch["heatmaps"].to(device), batch["weight"].to(device)).item()
            )

            coords, _ = predict_original_coords(
                pred.cpu().numpy(),
                batch["center"].numpy(),
                batch["size"].numpy(),
                args.input_w,
                args.input_h,
                stride,
            )
            gt = batch["kpts"].numpy()
            box_side = batch["bbox"].numpy()[:, 2:].max(axis=1)
            dist = np.linalg.norm(coords - gt[:, :, :2], axis=-1)
            norm_errors.append(dist / box_side[:, None])
            labeled.append(gt[:, :, 2] > 0)

    norm_errors = np.concatenate(norm_errors)
    labeled = np.concatenate(labeled)

    metrics = {"val_loss": float(np.mean(losses))}
    for group, group_slice in list(GROUP_SLICES.items()) + [("all", slice(None))]:
        err = norm_errors[:, group_slice][labeled[:, group_slice]]
        if err.size == 0:
            continue
        metrics[f"{group}_nme"] = float(err.mean())
        for threshold in PCK_THRESHOLDS:
            metrics[f"{group}_pck@{threshold}"] = float((err < threshold).mean())
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stride = 4

    train_samples = load_samples(args.train_ann, args.train_img_dir)
    if args.val_ann:
        val_samples = load_samples(args.val_ann, args.val_img_dir or args.train_img_dir)
    else:
        train_samples, val_samples = split_by_image_id(train_samples)

    print(f"Device: {device}")
    print(f"Train person instances: {len(train_samples)}")
    print(f"Val person instances: {len(val_samples)}")

    dataset_kwargs = {"input_h": args.input_h, "input_w": args.input_w, "stride": stride, "sigma": args.sigma}
    train_loader = DataLoader(
        CocoWholeBodyKeypoints(train_samples, train=True, **dataset_kwargs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        CocoWholeBodyKeypoints(val_samples, train=False, **dataset_kwargs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = KeypointHeatmapNet(NUM_KEYPOINTS, args.backbone, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    milestones = [int(args.epochs * 0.7), int(args.epochs * 0.9)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    log_rows = []
    best_pck = -1.0
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_pck = checkpoint.get("best_pck", best_pck)
        if os.path.isfile(log_path):
            with open(log_path, "r", newline="", encoding="utf-8") as f:
                log_rows = [row for row in csv.DictReader(f) if int(row["epoch"]) < start_epoch]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start = time.time()
        train_losses = []
        for step, batch in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            with autocast(device, not args.no_amp):
                pred = model(batch["image"].to(device, non_blocking=True))
            loss = weighted_mse(
                pred.float(),
                batch["heatmaps"].to(device, non_blocking=True),
                batch["weight"].to(device, non_blocking=True),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()
        train_seconds = time.time() - start

        metrics = evaluate(model, val_loader, device, args, stride, args.max_eval_batches)
        metrics = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **metrics}
        metrics["train_steps"] = len(train_losses)
        metrics["train_seconds"] = round(train_seconds, 1)
        metrics["seconds"] = round(time.time() - start, 1)
        log_rows.append(metrics)

        print(
            f"Epoch {epoch}: train_loss {metrics['train_loss']:.6f} val_loss {metrics['val_loss']:.6f} "
            f"PCK@0.05 {metrics['all_pck@0.05']:.3f} PCK@0.1 {metrics['all_pck@0.1']:.3f} "
            f"NME {metrics['all_nme']:.4f} ({metrics['seconds']}s)"
        )

        if metrics["all_pck@0.1"] > best_pck:
            best_pck = metrics["all_pck@0.1"]
            is_best = True
        else:
            is_best = False

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "metrics": metrics,
            "best_pck": best_pck,
        }
        torch.save(checkpoint, os.path.join(args.out_dir, "last.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(args.out_dir, "best.pt"))

        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    with open(os.path.join(args.out_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(log_rows[-1], f, indent=2)

    print("Per-group metrics (last epoch):")
    for key, value in log_rows[-1].items():
        if "pck" in key or "nme" in key:
            print(f"  {key}: {float(value):.4f}")
    print(f"Checkpoints written to: {args.out_dir}")


if __name__ == "__main__":
    main()
