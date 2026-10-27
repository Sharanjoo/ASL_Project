import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from label_points import GROUPS, LABEL_POINTS  # noqa: E402
from predict_wlasl import load_model, predict_clip  # noqa: E402

PCK_THRESHOLDS = (0.1, 0.2, 0.5)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score a Stage 1 checkpoint against hand-labeled WLASL frames (the proposal's domain-gap check)."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label_set", required=True, help="label_set.npz from make_label_set.py")
    parser.add_argument("--labels", nargs="+", required=True, help="One or more exported labels JSON files")
    parser.add_argument("--out", help="Optional JSON path for the metrics")
    return parser.parse_args()


def merge_labels(paths):
    merged, labelers = {}, []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        labelers.append(data.get("labeler", os.path.basename(path)))
        for key, points in data["labels"].items():
            merged.setdefault(key, {}).update(points)
    return merged, labelers


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_set = np.load(args.label_set)
    frames, keys = label_set["frames"], [str(k) for k in label_set["keys"]]
    labels, labelers = merge_labels(args.labels)

    model, train_args = load_model(args.checkpoint, device)
    preds = []
    for i in range(0, len(frames), 32):
        preds.append(predict_clip(model, frames[i : i + 32], train_args["input_h"], train_args["input_w"], device))
    preds = np.concatenate(preds)

    # errors[group] -> list of (pixel error / shoulder width), one per labeled point x model index
    errors = {group: [] for group in GROUPS}
    scored_frames = skipped_no_shoulders = unlabeled = 0
    for i, key in enumerate(keys):
        frame_labels = labels.get(key)
        if not frame_labels:
            unlabeled += 1
            continue
        left, right = frame_labels.get("left_shoulder"), frame_labels.get("right_shoulder")
        if not left or not right:
            skipped_no_shoulders += 1
            continue
        shoulder_width = float(np.hypot(left[0] - right[0], left[1] - right[1]))
        if shoulder_width < 5:
            skipped_no_shoulders += 1
            continue

        scored_frames += 1
        for name, group, indices, _hint in LABEL_POINTS:
            point = frame_labels.get(name)
            if not point:
                continue
            for idx in indices:
                error = np.hypot(preds[i, idx, 0] - point[0], preds[i, idx, 1] - point[1])
                errors[group].append(error / shoulder_width)

    results = {
        "checkpoint": args.checkpoint,
        "labelers": labelers,
        "frames_in_set": len(keys),
        "frames_scored": scored_frames,
        "frames_unlabeled": unlabeled,
        "frames_skipped_no_shoulders": skipped_no_shoulders,
        "groups": {},
    }
    all_errors = np.concatenate([np.array(e) for e in errors.values() if e]) if any(errors.values()) else np.array([])
    for group, values in list(errors.items()) + [("all", all_errors)]:
        values = np.array(values)
        if values.size == 0:
            continue
        results["groups"][group] = {
            "points": int(values.size),
            "mean_error": float(values.mean()),
            **{f"pck@{t}": float((values < t).mean()) for t in PCK_THRESHOLDS},
        }

    print(f"Labelers: {', '.join(labelers)}")
    print(
        f"Frames scored: {scored_frames} of {len(keys)} "
        f"({unlabeled} unlabeled, {skipped_no_shoulders} without both shoulders)"
    )
    print("Errors are a fraction of shoulder width.")
    print(f"{'group':12s} {'points':>6s} {'mean err':>9s} " + " ".join(f"{'PCK@' + str(t):>8s}" for t in PCK_THRESHOLDS))
    for group, m in results["groups"].items():
        print(
            f"{group:12s} {m['points']:6d} {m['mean_error']:9.3f} "
            + " ".join(f"{m[f'pck@{t}'] * 100:7.1f}%" for t in PCK_THRESHOLDS)
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics written to {args.out}")


if __name__ == "__main__":
    main()
