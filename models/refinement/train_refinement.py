import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from refiner import TemporalRefiner, refinement_loss
from windows import WINDOW_RADIUS, load_keypoint_dir, make_windows, synthetic_corruption, to_model_space


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: train temporal keypoint refinement (clean-frame keypoints are the target)."
    )
    parser.add_argument("--clean", required=True, help="manifest_keypoints.csv for clean frames (the target)")
    parser.add_argument(
        "--noisy", nargs="*", default=[],
        help="manifest_keypoints.csv files from degraded frames (used by --mode degraded)",
    )
    parser.add_argument(
        "--mode", choices=["synthetic", "degraded"], required=True,
        help="synthetic = keypoint-space corruption only (Baseline B); degraded = visual degradations (Proposed)",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--arch", choices=["transformer", "gru"], default="transformer")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--score_threshold", type=float, default=0.2, help="Only clean points at or above this score are trusted targets")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_pairs(clean, sources, mode, rng):
    """Returns (windows, targets) arrays for every frame of every clip and source."""
    windows, targets = [], []
    for video_id, target in clean.items():
        target = to_model_space(target)
        if mode == "synthetic":
            inputs = [synthetic_corruption(target, rng), target]
        else:
            inputs = [to_model_space(source[video_id]) for source in sources if video_id in source] + [target]
        for sequence in inputs:
            windows.append(make_windows(sequence))
            targets.append(target)
    return np.concatenate(windows), np.concatenate(targets)


def mean_trusted_error(pred, target, score_threshold):
    trusted = target[..., 2] >= score_threshold
    error = np.linalg.norm(pred[..., :2] - target[..., :2], axis=-1)
    return float(error[trusted].mean()) if trusted.any() else float("nan")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "degraded" and not args.noisy:
        raise SystemExit("--mode degraded needs at least one --noisy manifest")

    clean = {split: load_keypoint_dir(args.clean, split)[0] for split in ("train", "val")}
    noisy = {split: [load_keypoint_dir(path, split)[0] for path in args.noisy] for split in ("train", "val")}

    # Validation always measures recovery from the visually degraded inputs when they exist,
    # so synthetic- and degraded-trained refiners are compared on the same frames.
    val_mode = "degraded" if args.noisy else "synthetic"
    val_windows, val_targets = build_pairs(clean["val"], noisy["val"], val_mode, np.random.default_rng(1))
    val_windows_t = torch.from_numpy(val_windows).to(device)
    identity_error = mean_trusted_error(val_windows[:, WINDOW_RADIUS], val_targets, args.score_threshold)

    print(f"Mode: {args.mode} (validation on {val_mode} inputs)")
    print(f"Train clips: {len(clean['train'])}, val windows: {len(val_windows)}")
    print(f"Val error with no refinement (identity): {identity_error:.4f}")

    model = TemporalRefiner(window=2 * WINDOW_RADIUS + 1, arch=args.arch, hidden=args.hidden, num_layers=args.num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log_rows = []
    best_error = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        # Synthetic corruption is redrawn every epoch; degraded inputs are fixed files.
        train_windows, train_targets = build_pairs(clean["train"], noisy["train"], args.mode, rng)
        order = rng.permutation(len(train_windows))

        model.train()
        losses = []
        for i in range(0, len(order), args.batch_size):
            batch = order[i : i + args.batch_size]
            pred = model(torch.from_numpy(train_windows[batch]).to(device))
            loss = refinement_loss(pred, torch.from_numpy(train_targets[batch]).to(device), args.score_threshold)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = torch.cat(
                [model(val_windows_t[i : i + 4096]) for i in range(0, len(val_windows_t), 4096)]
            ).cpu().numpy()
        val_error = mean_trusted_error(val_pred, val_targets, args.score_threshold)
        log_rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_error": val_error})

        if val_error < best_error:
            best_error = val_error
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, os.path.join(args.out_dir, "best.pt"))

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}: train_loss {log_rows[-1]['train_loss']:.4f} val_error {val_error:.4f} ({time.time() - start:.1f}s)")

    with open(os.path.join(args.out_dir, "train_log.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    results = {"identity_val_error": identity_error, "best_val_error": best_error, "validation_inputs": val_mode}
    with open(os.path.join(args.out_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Best val error: {best_error:.4f} (identity {identity_error:.4f}); errors are in frame-normalised units")


if __name__ == "__main__":
    main()
