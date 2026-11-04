"""Run a third-party keypoint estimator over every clip under clean + the 9 degradations, writing the same
<out_root>/<condition>/<video_id>.npy + manifest_keypoints.csv layout that predict_wlasl.py writes for Stage 1.

The degraded frames are produced exactly as in predict_wlasl.py (same degrade_clip, same per-clip seeded rng,
occluders placed on the Stage 1 clean wrists), so every model sees identical damage.

Run from the repo root, with the estimator's own venv:
  .venv_mp/Scripts/python  models/baselines/extract_baseline.py --model mediapipe --workers 12
  .venv_rtm/Scripts/python models/baselines/extract_baseline.py --model rtmw --device cuda --workers 2
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "data", "degradation"))
from common import ESTIMATORS  # noqa: E402
from degrade import clip_rng, degrade_clip  # noqa: E402

CONDITIONS = ["clean"] + [f"{d}_{s}" for d in ("occlusion", "blur", "missing") for s in ("mild", "moderate", "severe")]

_estimator = None


def _init(model, kwargs):
    global _estimator
    _estimator = ESTIMATORS[model](**kwargs)


def _work(task):
    video_id, frames_path, condition, hand_path, seed, out_path = task
    if os.path.isfile(out_path):
        return video_id, 0.0
    frames = np.load(frames_path)
    if condition != "clean":
        degradation, severity = condition.split("_")
        hand = np.load(hand_path) if hand_path and os.path.isfile(hand_path) else None
        frames, _ = degrade_clip(frames, degradation, severity, clip_rng(video_id, degradation, severity, seed), hand)
    t0 = time.time()
    np.save(out_path, _estimator(frames))
    return video_id, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(ESTIMATORS))
    p.add_argument("--manifest", default="subset_100/frames16/manifest_frames.csv")
    p.add_argument("--hand_keypoints_dir", default="runs/pipeline/keypoints/clean")
    p.add_argument("--out_root", help="default runs/baselines/<model>/keypoints")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, help="only the first N clips (smoke test)")
    p.add_argument("--device", default="cpu", help="rtmw only: cpu or cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_root = args.out_root or os.path.join("runs", "baselines", args.model, "keypoints")
    with open(args.manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames) + ["keypoints_path"]
    if args.limit:
        rows = rows[: args.limit]

    kwargs = {"device": args.device} if args.model == "rtmw" else {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.model, kwargs)) as pool:
        for condition in args.conditions:
            out_dir = os.path.join(out_root, condition)
            os.makedirs(out_dir, exist_ok=True)
            tasks = [
                (r["video_id"], r["frames_path"], condition,
                 os.path.join(args.hand_keypoints_dir, f"{r['video_id']}.npy"), args.seed,
                 os.path.join(out_dir, f"{r['video_id']}.npy"))
                for r in rows
            ]
            t0 = time.time()
            for done, _ in enumerate(pool.imap_unordered(_work, tasks, chunksize=4), 1):
                if done % 250 == 0:
                    print(f"  {condition}: {done}/{len(tasks)} clips ({time.time() - t0:.0f}s)", flush=True)
            with open(os.path.join(out_dir, "manifest_keypoints.csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows({**r, "keypoints_path": os.path.join(out_dir, f"{r['video_id']}.npy")} for r in rows)
            print(f"{condition}: {len(rows)} clips in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
