"""Re-score MediaPipe keypoints to test how much its (placeholder) confidence scores matter.

Reads runs/baselines/mediapipe/keypoints and writes two variants next to it, same layout as extract_baseline.py:
  mediapipe_vis   hand points take the score of their wrist from the pose model (its visibility); mouth points take the
                  nose's visibility. Undetected points stay at 0. Body scores are unchanged.
  mediapipe_flat  every detected point (score > 0) gets 1.0, so no visibility information reaches Stage 3.
Positions are untouched; only the score channel changes.
"""
import csv
import os
import shutil

import numpy as np

SRC = os.path.join("runs", "baselines", "mediapipe", "keypoints")
MOUTH, LEFT_HAND, RIGHT_HAND = slice(11, 31), slice(31, 52), slice(52, 73)
LEFT_WRIST, RIGHT_WRIST, NOSE = 9, 10, 0


def vis(a):
    out = a.copy()
    det = a[..., 2] > 0
    for sl, src in ((LEFT_HAND, LEFT_WRIST), (RIGHT_HAND, RIGHT_WRIST), (MOUTH, NOSE)):
        out[:, sl, 2] = np.where(det[:, sl], a[:, src, 2][:, None], 0.0)
    return out


def flat(a):
    out = a.copy()
    out[..., 2] = (a[..., 2] > 0).astype(np.float32)
    return out


for name, fn in (("mediapipe_vis", vis), ("mediapipe_flat", flat)):
    dst_root = os.path.join("runs", "baselines", name, "keypoints")
    for cond in sorted(os.listdir(SRC)):
        src_dir, dst_dir = os.path.join(SRC, cond), os.path.join(dst_root, cond)
        if not os.path.isfile(os.path.join(src_dir, "manifest_keypoints.csv")):
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if f.endswith(".npy"):
                np.save(os.path.join(dst_dir, f), fn(np.load(os.path.join(src_dir, f))).astype(np.float32))
        with open(os.path.join(src_dir, "manifest_keypoints.csv"), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = reader.fieldnames
        for r in rows:
            r["keypoints_path"] = os.path.join(dst_dir, os.path.basename(r["keypoints_path"]))
        with open(os.path.join(dst_dir, "manifest_keypoints.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print("wrote", dst_root, flush=True)
