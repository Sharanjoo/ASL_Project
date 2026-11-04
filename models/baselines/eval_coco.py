"""COCO-WholeBody val check for Stage 1 and the baselines, using Stage 1's own validation protocol
(train_keypoints.py evaluate): person crop from the GT box at 256x192, error / max(box w, h), labeled points only,
PCK@0.05 and PCK@0.1 per group. Undetected points count as misses (they are placed at the crop origin).

  .venv/Scripts/python     models/baselines/eval_coco.py --model stage1 --checkpoint runs/keypoints_resnet50_256x192/best.pt
  .venv_mp/Scripts/python  models/baselines/eval_coco.py --model mediapipe --workers 12
  .venv_rtm/Scripts/python models/baselines/eval_coco.py --model rtmw --device cuda --workers 2
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KP = os.path.join(HERE, "..", "keypoints")
sys.path.insert(0, HERE)
sys.path.insert(0, KP)
from keypoint_spec import GROUP_SLICES, subset_keypoints  # noqa: E402
from transforms import apply_affine, box_to_center_size, get_affine  # noqa: E402

INPUT_H, INPUT_W = 256, 192
PCK_THRESHOLDS = (0.05, 0.1)


def load_samples(ann_path, img_dir, min_visible=8, min_box_side=48):
    """Same filters as models/keypoints/coco_wholebody_dataset.load_samples (without the torch import)."""
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = {im["id"]: im["file_name"] for im in data["images"]}
    samples = []
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        if min(w, h) < min_box_side:
            continue
        kpts = subset_keypoints(ann)
        if (kpts[:, 2] > 0).sum() < min_visible:
            continue
        samples.append({"img_path": os.path.join(img_dir, names[ann["image_id"]]),
                        "bbox": np.array(ann["bbox"], np.float32), "kpts": kpts})
    return samples


def crop_person(sample):
    img = cv2.cvtColor(cv2.imread(sample["img_path"]), cv2.COLOR_BGR2RGB)
    center, size = box_to_center_size(sample["bbox"], INPUT_W / INPUT_H)
    matrix = get_affine(center, size, 0.0, INPUT_W, INPUT_H)
    crop = cv2.warpAffine(img, matrix, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
    return crop, get_affine(center, size, 0.0, INPUT_W, INPUT_H, inverse=True)


_estimator = None


def _init(model, kwargs):
    global _estimator
    from common import ESTIMATORS
    _estimator = ESTIMATORS[model](**kwargs)


def _work(sample):
    crop, inverse = crop_person(sample)
    pred = _estimator(crop[None])[0]
    return apply_affine(pred[:, :2], inverse)


def predict_stage1(samples, checkpoint, batch=64):
    import torch
    from predict_wlasl import load_model
    from transforms import decode_heatmaps, normalize_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint, device)
    out = []
    for i in range(0, len(samples), batch):
        crops, inverses = zip(*[crop_person(s) for s in samples[i:i + batch]])
        x = torch.from_numpy(np.stack([normalize_image(c) for c in crops])).to(device)
        with torch.no_grad():
            coords, _ = decode_heatmaps(model(x).float().cpu().numpy(), 4)
        out += [apply_affine(c, inv) for c, inv in zip(coords, inverses)]
    return np.stack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["stage1", "mediapipe", "rtmw"])
    p.add_argument("--checkpoint", default="runs/keypoints_resnet50_256x192/best.pt")
    p.add_argument("--val_ann", default="coco_wholebody_ann/coco_wholebody_val_v1.0.json")
    p.add_argument("--val_img_dir", default="coco_images/val2017")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--limit", type=int)
    p.add_argument("--out_dir")
    args = p.parse_args()

    samples = load_samples(args.val_ann, args.val_img_dir)
    if args.limit:
        samples = samples[: args.limit]
    print(f"{args.model}: {len(samples)} person instances", flush=True)

    t0 = time.time()
    if args.model == "stage1":
        preds = predict_stage1(samples, args.checkpoint)
    else:
        kwargs = {"device": args.device} if args.model == "rtmw" else {}
        with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.model, kwargs)) as pool:
            preds = np.stack(list(pool.imap(_work, samples, chunksize=16)))
    print(f"inference {time.time() - t0:.0f}s", flush=True)

    gt = np.stack([s["kpts"] for s in samples])
    box_side = np.array([max(s["bbox"][2], s["bbox"][3]) for s in samples])
    err = np.linalg.norm(preds - gt[:, :, :2], axis=-1) / box_side[:, None]
    labeled = gt[:, :, 2] > 0

    metrics = {"model": args.model, "instances": len(samples)}
    for group, sl in list(GROUP_SLICES.items()) + [("all", slice(None))]:
        e = err[:, sl][labeled[:, sl]]
        metrics[f"{group}_nme"] = float(e.mean())
        for t in PCK_THRESHOLDS:
            metrics[f"{group}_pck@{t}"] = float((e < t).mean())
    out_dir = args.out_dir or os.path.join("runs", "baselines", args.model)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "coco_val_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"{'group':12s} {'NME':>7s} " + " ".join(f"PCK@{t:<5}" for t in PCK_THRESHOLDS))
    for g in list(GROUP_SLICES) + ["all"]:
        print(f"{g:12s} {metrics[f'{g}_nme']:7.3f} " + " ".join(f"{metrics[f'{g}_pck@{t}'] * 100:8.1f}%" for t in PCK_THRESHOLDS))


if __name__ == "__main__":
    main()
