import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

from keypoint_spec import GROUP_COLORS, GROUP_SLICES, NUM_KEYPOINTS
from model import KeypointHeatmapNet
from transforms import apply_affine, box_to_center_size, decode_heatmaps, get_affine, normalize_image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "degradation"))
from degrade import DEGRADATIONS, SEVERITIES, clip_rng, degrade_clip  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Stage 1 keypoint extractor over sampled WLASL frame arrays."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="manifest_frames.csv from sample_frames.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--overlay_count", type=int, default=4)
    parser.add_argument("--degradation", choices=DEGRADATIONS, default="clean")
    parser.add_argument("--severity", choices=SEVERITIES, default="moderate")
    parser.add_argument(
        "--hand_keypoints_dir",
        help="Clean-run keypoint dir; its wrist positions place the occluder for --degradation occlusion",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    model = KeypointHeatmapNet(NUM_KEYPOINTS, train_args["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), train_args


def predict_clip(model, frames, input_h, input_w, device, stride=4):
    frame_h, frame_w = frames.shape[1:3]
    center, size = box_to_center_size([0, 0, frame_w, frame_h], input_w / input_h, padding=1.0)
    matrix = get_affine(center, size, 0.0, input_w, input_h)
    inverse = get_affine(center, size, 0.0, input_w, input_h, inverse=True)

    crops = [cv2.warpAffine(frame, matrix, (input_w, input_h), flags=cv2.INTER_LINEAR) for frame in frames]
    batch = torch.from_numpy(np.stack([normalize_image(c) for c in crops])).to(device)

    with torch.no_grad():
        heatmaps = model(batch).cpu().numpy()

    coords, scores = decode_heatmaps(heatmaps, stride)
    coords = np.stack([apply_affine(c, inverse) for c in coords])
    return np.concatenate([coords, scores[..., None]], axis=-1).astype(np.float32)


def draw_overlay(frames, keypoints, score_threshold=0.2):
    panels = []
    for frame, kpts in zip(frames, keypoints):
        panel = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
        for group, group_slice in GROUP_SLICES.items():
            for x, y, score in kpts[group_slice]:
                if score >= score_threshold:
                    cv2.circle(panel, (int(round(x)), int(round(y))), 2, GROUP_COLORS[group], -1)
        panels.append(panel)
    return np.hstack(panels)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, train_args = load_model(args.checkpoint, device)

    with open(args.manifest, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames) + ["keypoints_path"]

    out_rows = []
    overlays_written = 0
    for row in rows:
        frames = np.load(row["frames_path"])
        if args.degradation != "clean":
            hand_keypoints = None
            if args.hand_keypoints_dir:
                hand_path = os.path.join(args.hand_keypoints_dir, f"{row['video_id']}.npy")
                if os.path.isfile(hand_path):
                    hand_keypoints = np.load(hand_path)
            rng = clip_rng(row["video_id"], args.degradation, args.severity, args.seed)
            frames, _ = degrade_clip(frames, args.degradation, args.severity, rng, hand_keypoints)

        keypoints = predict_clip(model, frames, train_args["input_h"], train_args["input_w"], device)

        keypoints_path = os.path.join(args.out_dir, f"{row['video_id']}.npy")
        np.save(keypoints_path, keypoints)
        out_rows.append({**row, "keypoints_path": keypoints_path})

        if overlays_written < args.overlay_count:
            overlay_path = os.path.join(args.out_dir, f"overlay_{row['video_id']}_{row['gloss']}.png")
            cv2.imwrite(overlay_path, draw_overlay(frames, keypoints))
            overlays_written += 1

    manifest_path = os.path.join(args.out_dir, "manifest_keypoints.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    condition = "clean" if args.degradation == "clean" else f"{args.degradation}/{args.severity}"
    print(f"Condition: {condition}")
    print(f"Clips processed: {len(out_rows)}")
    print(f"Keypoint array shape per clip: (num_frames, {NUM_KEYPOINTS}, 3) as (x, y, score)")
    print(f"Overlays written: {overlays_written}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
