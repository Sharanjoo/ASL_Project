import argparse
import base64
import csv
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np

from label_points import LABEL_POINTS

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP_COLORS = {"upper_body": "#1a9e3f", "mouth": "#e07b00", "left_hand": "#2563eb", "right_hand": "#dc2626"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pick WLASL frames for hand-labeling and write a self-contained labeling tool."
    )
    parser.add_argument("--manifest", required=True, help="manifest_frames.csv (e.g. subset_100/frames16)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--frame_index", type=int, default=-1, help="Frame within each clip; -1 = middle frame")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def pick_rows(rows, count, seed):
    """Round-robin across signers so no one signer dominates the label set."""
    rng = random.Random(seed)
    by_signer = defaultdict(list)
    for row in rows:
        by_signer[row["signer_id"]].append(row)
    for signer_rows in by_signer.values():
        rng.shuffle(signer_rows)
    signers = sorted(by_signer)
    rng.shuffle(signers)

    picked = []
    while len(picked) < count and any(by_signer.values()):
        for signer in signers:
            if by_signer[signer] and len(picked) < count:
                picked.append(by_signer[signer].pop())
    return picked


def png_data_uri(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.manifest, "r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == args.split]
    picked = pick_rows(rows, args.count, args.seed)

    frames, entries = [], []
    for row in picked:
        clip = np.load(row["frames_path"])
        t = clip.shape[0] // 2 if args.frame_index < 0 else min(args.frame_index, clip.shape[0] - 1)
        frames.append(clip[t])
        entries.append(
            {
                "key": f"{row['video_id']}:{t}",
                "video_id": row["video_id"],
                "frame": t,
                "gloss": row["gloss"],
                "signer_id": row["signer_id"],
            }
        )

    set_id = f"wlasl-{args.split}-{len(entries)}-seed{args.seed}"
    np.savez_compressed(
        os.path.join(args.out_dir, "label_set.npz"),
        frames=np.stack(frames),
        keys=np.array([e["key"] for e in entries]),
    )
    with open(os.path.join(args.out_dir, "label_set.json"), "w", encoding="utf-8") as f:
        json.dump({"set_id": set_id, "frames": entries}, f, indent=1)

    payload = {
        "set_id": set_id,
        "points": [
            {"name": name, "group": group, "hint": hint, "color": GROUP_COLORS[group]}
            for name, group, _indices, hint in LABEL_POINTS
        ],
        "frames": [{**e, "png": png_data_uri(img)} for e, img in zip(entries, frames)],
    }
    with open(os.path.join(HERE, "label_tool_template.html"), "r", encoding="utf-8") as f:
        html = f.read().replace("/*__LABEL_SET__*/null", json.dumps(payload))
    tool_path = os.path.join(args.out_dir, "label_tool.html")
    with open(tool_path, "w", encoding="utf-8") as f:
        f.write(html)

    signers = len({e["signer_id"] for e in entries})
    glosses = len({e["gloss"] for e in entries})
    print(f"Frames picked: {len(entries)} from the {args.split} split ({signers} signers, {glosses} glosses)")
    print(f"Points per frame: {len(LABEL_POINTS)}")
    print(f"Labeling tool: {tool_path}")
    print(f"Frames for scoring: {os.path.join(args.out_dir, 'label_set.npz')}")


if __name__ == "__main__":
    main()
