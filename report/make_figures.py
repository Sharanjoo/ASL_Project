import argparse
import csv
import json
import os
import sys
from collections import Counter

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data", "degradation"))
sys.path.insert(0, os.path.join(ROOT, "models", "keypoints"))

from degrade import clip_rng, degrade_clip  # noqa: E402
from keypoint_spec import GROUP_SLICES, subset_keypoints  # noqa: E402

# Reference categorical palette, light mode (slots 1-3 validate all-pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MUTED_FILL = "#d9d8d3"
GROUP_COLORS = {"upper_body": "#008300", "mouth": "#eb6834", "left_hand": "#2a78d6", "right_hand": "#e34948"}
GROUP_LABELS = {"upper_body": "Upper body (11)", "mouth": "Mouth (20)", "left_hand": "Left hand (21)", "right_hand": "Right hand (21)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT,
    "axes.labelcolor": TEXT_SECONDARY, "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
})


def parse_args():
    parser = argparse.ArgumentParser(description="Build the report's data and method figures.")
    parser.add_argument("--manifest", default=os.path.join(ROOT, "subset_100", "manifest.csv"))
    parser.add_argument("--frames_manifest", default=os.path.join(ROOT, "subset_100", "frames16", "manifest_frames.csv"))
    parser.add_argument("--coco_ann", default=os.path.join(ROOT, "coco_wholebody_ann", "coco_wholebody_val_v1.0.json"))
    parser.add_argument("--coco_img_dir", default=os.path.join(ROOT, "coco_images", "val2017"))
    parser.add_argument("--coco_image_id", type=int, help="Force the keypoint-layout example to this val2017 image")
    parser.add_argument("--degrade_clip", default="00618", help="video_id used for the degradation figure")
    parser.add_argument(
        "--hand_keypoints_dir",
        help="Clean keypoints from a trained extractor (e.g. runs/pipeline/keypoints/clean); "
        "places the occluders on the wrists. Without it they use the fallback hand position.",
    )
    parser.add_argument("--out_dir", default=os.path.join(HERE, "figures"))
    return parser.parse_args()


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def coverage_figure(manifest_rows, usable_rows, out_path):
    total = Counter(r["gloss"] for r in manifest_rows)
    usable = {s: Counter(r["gloss"] for r in usable_rows if r["split"] == s) for s in ("train", "val", "test")}
    order = sorted(total, key=lambda g: (-sum(usable[s][g] for s in usable), g))
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=200)
    ax.bar(x, [total[g] for g in order], width=0.8, color=MUTED_FILL, label="In WLASL metadata", zorder=1)
    bottom = np.zeros(len(order))
    for color, split in zip(SERIES, ("train", "val", "test")):
        values = np.array([usable[split][g] for g in order])
        ax.bar(x, values, width=0.8, bottom=bottom, color=color, label=f"Usable — {split}",
               edgecolor="white", linewidth=0.6, zorder=2)
        bottom += values

    ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_xticks([])
    ax.set_xlabel("100 glosses, sorted by usable clips")
    ax.set_ylabel("Clips per gloss")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    used = len(usable_rows)
    ax.set_title(f"WLASL100 coverage: {used:,} of {len(manifest_rows):,} clips usable ({used / len(manifest_rows):.0%})",
                 loc="left", fontsize=11, color=TEXT, pad=26)
    ax.legend(frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def degradation_figure(frames_path, video_id, out_path, hand_keypoints_dir=None, every=3):
    frames = np.load(frames_path)
    hand_keypoints = None
    if hand_keypoints_dir:
        hand_keypoints = np.load(os.path.join(hand_keypoints_dir, f"{video_id}.npy"))
    rows = [("Clean", frames)]
    for degradation, title in (("occlusion", "Hand occlusion"), ("blur", "Motion blur"), ("missing", "Missing frames")):
        rng = clip_rng(video_id, degradation, "moderate")
        degraded, _ = degrade_clip(frames, degradation, "moderate", rng, hand_keypoints)
        rows.append((f"{title} (moderate)", degraded))

    shown = list(range(0, frames.shape[0], every))
    fig, axes = plt.subplots(len(rows), len(shown), figsize=(1.6 * len(shown), 1.75 * len(rows)), dpi=200)
    for r, (title, clip) in enumerate(rows):
        for c, t in enumerate(shown):
            ax = axes[r, c]
            ax.imshow(clip[t])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(f"frame {t + 1}", fontsize=8, color=TEXT_SECONDARY)
        axes[r, 0].set_ylabel(title, fontsize=9, color=TEXT, rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle("Controlled degradations at moderate severity (50% of frames hit)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def keypoint_layout_figure(ann_path, img_dir, out_path, image_id=None):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_file = {im["id"]: im["file_name"] for im in data["images"]}

    # A large, frontal person with all four groups labeled and the hands clear of each other
    # and of the mouth, so every group reads separately.
    best, best_area = None, 0
    for a in data["annotations"]:
        if image_id is not None and a["image_id"] != image_id:
            continue
        if a.get("iscrowd") or not (a.get("lefthand_valid") and a.get("righthand_valid") and a.get("face_valid")):
            continue
        kpts = subset_keypoints(a)
        if (kpts[:, 2] > 0).sum() < 70:
            continue
        w = a["bbox"][2]
        left, right = kpts[GROUP_SLICES["left_hand"], :2].mean(0), kpts[GROUP_SLICES["right_hand"], :2].mean(0)
        mouth = kpts[GROUP_SLICES["mouth"], :2].mean(0)
        frontal = kpts[5, 0] > kpts[6, 0]  # signer's left shoulder on the image right
        apart = np.linalg.norm(left - right) > 0.35 * w
        clear_of_mouth = min(np.linalg.norm(left - mouth), np.linalg.norm(right - mouth)) > 0.2 * w
        adult = a.get("face_box") and a["face_box"][3] < 0.22 * a["bbox"][3]  # adult head-to-body ratio
        if not (frontal and apart and clear_of_mouth and adult):
            continue
        area = a["bbox"][2] * a["bbox"][3]
        if area > best_area:
            best, best_area = a, area

    img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, id_to_file[best["image_id"]])), cv2.COLOR_BGR2RGB)
    x, y, w, h = best["bbox"]
    pad = 0.08
    x0, y0 = int(max(x - pad * w, 0)), int(max(y - pad * h, 0))
    x1, y1 = int(min(x + w * (1 + pad), img.shape[1])), int(min(y + h * (1 + pad), img.shape[0]))
    kpts = subset_keypoints(best)

    fig, ax = plt.subplots(figsize=(4.4, 4.4 * (y1 - y0) / (x1 - x0)), dpi=200)
    ax.imshow(img[y0:y1, x0:x1])
    for group, s in GROUP_SLICES.items():
        pts = kpts[s]
        pts = pts[pts[:, 2] > 0]
        ax.scatter(pts[:, 0] - x0, pts[:, 1] - y0, s=9, color=GROUP_COLORS[group], edgecolors="white",
                   linewidths=0.4, label=GROUP_LABELS[group], zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=True, framealpha=0.9, fontsize=7, loc="lower left", markerscale=1.6)
    ax.set_title("73 signing keypoints (COCO-WholeBody ground truth)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return best["image_id"]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    manifest_rows = read_csv(args.manifest)
    usable_rows = read_csv(args.frames_manifest)
    coverage_figure(manifest_rows, usable_rows, os.path.join(args.out_dir, "fig_dataset_coverage.png"))

    frames_dir = os.path.dirname(args.frames_manifest)
    degradation_figure(os.path.join(frames_dir, f"{args.degrade_clip}.npy"), args.degrade_clip,
                       os.path.join(args.out_dir, "fig_degradations.png"), args.hand_keypoints_dir)
    if not args.hand_keypoints_dir:
        print("Note: occluders use the fallback hand position; re-run with --hand_keypoints_dir "
              "after the GPU extractor is trained so they sit on the wrists.")

    image_id = keypoint_layout_figure(args.coco_ann, args.coco_img_dir, os.path.join(args.out_dir, "fig_keypoint_layout.png"),
                                      args.coco_image_id)

    print(f"Figures written to {args.out_dir}")
    print(f"Keypoint layout example: COCO val2017 image {image_id}")


if __name__ == "__main__":
    main()
