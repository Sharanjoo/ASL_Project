"""Build the folder a teammate takes to the Mobilint NPU lab machine. See models/npu/README_NPU.md.

    python models/npu/make_npu_bundle.py --export_dir runs/npu_export --out_dir ../asl_npu_bundle

Inputs prepared here so the lab machine needs no PyTorch, COCO download or WLASL videos:
  calib_data/          300 PNG crops at the model's input size for INT8 calibration: 150 COCO val2017 people
                       NOT in the COCO evaluation subset, plus 150 frames from WLASL TRAIN-split clips
  data/coco_eval.npz   COCO val2017 people with image_id % 4 == 0: input crops, ground truth, crop->image transform
  data/label_set.npz   the 100 hand-labeled domain-gap frames (scored back on the laptop)
  data/wlasl/          the 161 WLASL100 test clips (16 frames each), their gloss/split, and the FP32 clean-run
                       keypoints, which place the occluder exactly where the FP32 robustness run placed it
"""
import argparse
import csv
import glob
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "models", "keypoints"))

from coco_wholebody_dataset import load_samples  # noqa: E402
from transforms import box_to_center_size, get_affine  # noqa: E402

INPUT_H, INPUT_W = 256, 192
LIB_FILES = ["models/keypoints/transforms.py", "models/keypoints/keypoint_spec.py", "data/degradation/degrade.py"]


def coco_crop(sample):
    img = cv2.cvtColor(cv2.imread(sample["img_path"]), cv2.COLOR_BGR2RGB)
    center, size = box_to_center_size(sample["bbox"], INPUT_W / INPUT_H)
    matrix = get_affine(center, size, 0.0, INPUT_W, INPUT_H)
    inverse = get_affine(center, size, 0.0, INPUT_W, INPUT_H, inverse=True)
    return cv2.warpAffine(img, matrix, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR), inverse


def frame_crop(frame):
    """Whole-frame crop at the model's aspect ratio, as predict_wlasl.predict_clip does."""
    h, w = frame.shape[:2]
    center, size = box_to_center_size([0, 0, w, h], INPUT_W / INPUT_H, padding=1.0)
    return cv2.warpAffine(frame, get_affine(center, size, 0.0, INPUT_W, INPUT_H), (INPUT_W, INPUT_H),
                          flags=cv2.INTER_LINEAR)


def save_png(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Build the Mobilint NPU bundle for the Stage 1 extractor.")
    parser.add_argument("--export_dir", required=True, help="Output of export_onnx.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--coco_ann", default="coco_wholebody_ann/coco_wholebody_val_v1.0.json")
    parser.add_argument("--coco_img_dir", default="coco_images/val2017")
    parser.add_argument("--wlasl_manifest", default="runs/pipeline/keypoints/clean/manifest_keypoints.csv",
                        help="FP32 clean-run keypoint manifest: frames, split and occluder positions")
    parser.add_argument("--label_set", default="domain_gap/label_set.npz")
    parser.add_argument("--wlasl_splits", nargs="+", default=["test"],
                        help="WLASL splits to bundle; train val test when Stages 2-3 will be retrained on NPU keypoints")
    parser.add_argument("--coco_holdout_mod", type=int, default=4)
    parser.add_argument("--calib_per_source", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.out_dir
    for sub in ("calib_data", "data/wlasl/frames", "data/wlasl/hand_keypoints", "lib"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    for name in ("model.onnx", "model_backbone.onnx", "model_head.onnx"):
        shutil.copy(os.path.join(args.export_dir, name), out)
    for path in LIB_FILES:
        shutil.copy(os.path.join(REPO, path), os.path.join(out, "lib"))
    for name in ("convert_to_mxq.py", "run_npu.py", "README_NPU.md"):
        shutil.copy(os.path.join(HERE, name), out)
    shutil.copy(args.label_set, os.path.join(out, "data", "label_set.npz"))

    # COCO: evaluation people and calibration people come from disjoint images.
    samples = load_samples(args.coco_ann, args.coco_img_dir)
    eval_samples = [s for s in samples if s["image_id"] % args.coco_holdout_mod == 0]
    calib_pool = [s for s in samples if s["image_id"] % args.coco_holdout_mod != 0]
    crops, inverses = zip(*(coco_crop(s) for s in eval_samples))
    np.savez_compressed(
        os.path.join(out, "data", "coco_eval.npz"),
        crops=np.stack(crops), inverse=np.stack(inverses).astype(np.float32),
        kpts=np.stack([s["kpts"] for s in eval_samples]).astype(np.float32),
        box_side=np.array([s["bbox"][2:].max() for s in eval_samples], dtype=np.float32),
        image_id=np.array([s["image_id"] for s in eval_samples]),
    )
    for i in rng.choice(len(calib_pool), args.calib_per_source, replace=False):
        crop, _ = coco_crop(calib_pool[i])
        save_png(os.path.join(out, "calib_data", f"coco_{calib_pool[i]['image_id']}_{i}.png"), crop)

    # WLASL: calibration frames from train clips; the test clips for evaluation.
    with open(args.wlasl_manifest, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train = [r for r in rows if r["split"] == "train"]
    for i in rng.choice(len(train), args.calib_per_source, replace=False):
        frames = np.load(train[i]["frames_path"])
        t = int(rng.integers(len(frames)))
        save_png(os.path.join(out, "calib_data", f"wlasl_{train[i]['video_id']}_{t}.png"), frame_crop(frames[t]))

    clips = [r for r in rows if r["split"] in args.wlasl_splits]
    for r in clips:
        shutil.copy(r["frames_path"], os.path.join(out, "data", "wlasl", "frames", f"{r['video_id']}.npy"))
        shutil.copy(r["keypoints_path"], os.path.join(out, "data", "wlasl", "hand_keypoints", f"{r['video_id']}.npy"))
    with open(os.path.join(out, "data", "wlasl", "clips.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "gloss", "split"])
        writer.writerows([[r["video_id"], r["gloss"], r["split"]] for r in clips])

    calib = len(glob.glob(os.path.join(out, "calib_data", "*.png")))
    print(f"Bundle written to {out}")
    print(f"  COCO evaluation people: {len(eval_samples)} (image_id % {args.coco_holdout_mod} == 0)")
    print(f"  calibration images:     {calib} (COCO from other images + WLASL train frames)")
    print(f"  WLASL clips:            {len(clips)} ({' '.join(args.wlasl_splits)})")


if __name__ == "__main__":
    main()
