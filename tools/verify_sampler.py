"""Check the rebuilt sample_frames.py against the original run's Stage 1 keypoints.

For each clip you have a video for, this re-samples frames with the rebuilt
sampler, runs Stage 1, and compares with the stored keypoints from the
original pipeline (runs/pipeline/keypoints/clean/<video_id>.npy).

Small differences (a few pixels on confident points) mean the rebuilt crop
matches the original. Large ones (tens of pixels) mean the crop or frame
spacing differs.

Usage (from the repo root):
  python tools/verify_sampler.py --video_dir check_videos
Put a few test videos in check_videos/, named <video_id>.mp4 (video_ids and
URLs are in the manifest; the direct .mp4 links are the easiest to download).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "keypoints"))
sys.path.insert(0, str(ROOT / "data" / "wlasl"))

from predict_wlasl import load_model, predict_clip  # noqa: E402
from sample_frames import find_video, sample_video  # noqa: E402


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="check_videos")
    ap.add_argument("--manifest", default="runs/pipeline/keypoints/clean/manifest_keypoints.csv")
    ap.add_argument("--checkpoint", default="runs/keypoints_resnet50_256x192/best.pt")
    ap.add_argument("--min_score", type=float, default=0.3, help="only compare confident points")
    args = ap.parse_args()

    device = pick_device()
    try:
        model, train_args = load_model(args.checkpoint, device)
    except Exception as exc:  # e.g. an op not supported on MPS
        print(f"[warn] loading on {device} failed ({exc}); using CPU")
        device = torch.device("cpu")
        model, train_args = load_model(args.checkpoint, device)
    print(f"device: {device}")

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    manifest_dir = Path(args.manifest).parent
    video_dir = Path(args.video_dir)

    results = []
    for row in rows:
        path = find_video(video_dir, row)
        if path is None:
            continue
        stored_path = manifest_dir / f"{row['video_id']}.npy"
        if not stored_path.exists():
            print(f"[skip] {row['video_id']}: no stored keypoints")
            continue
        clip = sample_video(path, 16, 224, "bbox", row.get("bbox"),
                            int(row.get("frame_start", 1) or 1), int(row.get("frame_end", -1) or -1))
        if clip is None:
            print(f"[skip] {row['video_id']}: could not decode")
            continue
        with torch.no_grad():
            new = predict_clip(model, clip, train_args["input_h"], train_args["input_w"], device)
        old = np.load(stored_path)
        mask = (old[..., 2] >= args.min_score) & (new[..., 2] >= args.min_score)
        if not mask.any():
            print(f"[skip] {row['video_id']}: no confident points to compare")
            continue
        err = np.linalg.norm(new[..., :2] - old[..., :2], axis=-1)[mask]
        results.append((row["video_id"], row["gloss"], float(np.median(err)), float(err.mean())))
        print(f"{row['video_id']} ({row['gloss']}): median {np.median(err):.1f} px, "
              f"mean {err.mean():.1f} px over {mask.sum()} points")

    if not results:
        print(f"No matching videos found in {video_dir}/ (name them <video_id>.mp4).")
        return 1
    med = np.median([r[2] for r in results])
    print(f"\n{len(results)} clips, overall median error {med:.1f} px (224x224 frame)")
    print("Verdict:", "matches the original crop" if med <= 3 else
          "close; small differences" if med <= 8 else
          "does NOT match; crop or frame spacing differs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
