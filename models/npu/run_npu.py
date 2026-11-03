r"""run_npu.py -- run the Stage 1 keypoint extractor on the Mobilint NPU (MACCEL) and save everything
the laptop needs to score it. No PyTorch needed: numpy, opencv-python and maccel (or onnxruntime).

    python run_npu.py all --mxq model.mxq                                   # the whole job, ~1 hour
    python run_npu.py all --mxq model_backbone.mxq --head_onnx model_head.onnx   # plan B (README step 3b)
    python run_npu.py all --backend onnx --onnx model.onnx                  # same job in float on the CPU

Steps (each can also be run on its own):
  check       loads the model, runs 32 COCO crops, works out the output layout, prints a quick PCK
  coco        PCK on the COCO val2017 evaluation people (compare with the FP32 numbers in README_NPU.md)
  speed       milliseconds per frame on the accelerator and end to end
  domain_gap  keypoints for the 100 hand-labeled WLASL frames (scored back on the laptop)
  wlasl       keypoints for the clips in data/wlasl/clips.csv, clean and under the 9 degradations
Results go to results_<backend>/ by default; finished WLASL clips are skipped on a re-run.
"""
import argparse
import csv
import json
import os
import platform
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(HERE, "lib"), os.path.join(HERE, "..", "keypoints"), os.path.join(HERE, "..", "..", "data", "degradation")):
    sys.path.insert(0, path)

from degrade import clip_rng, degrade_clip  # noqa: E402
from keypoint_spec import GROUP_SLICES, NUM_KEYPOINTS  # noqa: E402
from transforms import apply_affine, box_to_center_size, decode_heatmaps, get_affine  # noqa: E402

INPUT_H, INPUT_W, STRIDE = 256, 192, 4
HEATMAP_CHW = (NUM_KEYPOINTS, INPUT_H // STRIDE, INPUT_W // STRIDE)
CONDITIONS = ["clean"] + [f"{d}_{s}" for d in ("occlusion", "blur", "missing") for s in ("mild", "moderate", "severe")]


def unwrap(raw):
    while isinstance(raw, (list, tuple)):
        assert len(raw) == 1, f"expected one output, got {len(raw)}"
        raw = raw[0]
    return np.squeeze(np.asarray(raw, dtype=np.float32))


class Runner:
    """One preprocessed crop, (256, 192, 3) float32 RGB in [0, 1], -> (73, 64, 48) heatmaps."""

    def __init__(self, args):
        self.head = None
        if args.head_onnx:
            import onnxruntime as ort
            self.head = ort.InferenceSession(args.head_onnx, providers=["CPUExecutionProvider"])
            self.head_input = self.head.get_inputs()[0].name
            self.out_chw = tuple(self.head.get_inputs()[0].shape[1:])
        else:
            self.out_chw = HEATMAP_CHW

        if args.backend == "npu":
            import maccel
            self.accelerator = maccel.Accelerator(0)
            self.model = maccel.Model(args.mxq, maccel.ModelConfig())
            self.model.launch(self.accelerator)
            self.infer = lambda x: self.model.infer(x)
        else:
            import onnxruntime as ort
            session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
            name = session.get_inputs()[0].name
            self.infer = lambda x: session.run(None, {name: x[None]})[0]
        self.layout = args.layout

    def to_chw(self, raw, layout=None):
        out = unwrap(raw)
        c, h, w = self.out_chw
        assert out.size == c * h * w, f"accelerator returned {out.shape}, expected {self.out_chw} in some layout"
        layout = layout or self.layout
        if layout == "auto" and out.ndim == 3:
            layout = "chw" if out.shape == (c, h, w) else "hwc" if out.shape == (h, w, c) else "auto"
        if layout == "auto":
            raise RuntimeError("Output layout is ambiguous; run the 'check' step first.")
        return out.reshape(c, h, w) if layout == "chw" else out.reshape(h, w, c).transpose(2, 0, 1)

    def heatmaps(self, img01, layout=None):
        out = self.to_chw(self.infer(np.ascontiguousarray(img01, dtype=np.float32)), layout)
        if self.head is not None:
            out = self.head.run(None, {self.head_input: out[None]})[0][0]
        return out


def run_crops(runner, crops, log_every=0, layout=None):
    heatmaps = []
    for i, crop in enumerate(crops):
        heatmaps.append(runner.heatmaps(crop.astype(np.float32) / 255.0, layout))
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(crops)}", flush=True)
    return np.stack(heatmaps)


def predict_frames(runner, frames):
    """Whole-frame crops, as models/keypoints/predict_wlasl.py does. Returns (T, 73, 3) x, y, score."""
    h, w = frames.shape[1:3]
    center, size = box_to_center_size([0, 0, w, h], INPUT_W / INPUT_H, padding=1.0)
    matrix = get_affine(center, size, 0.0, INPUT_W, INPUT_H)
    inverse = get_affine(center, size, 0.0, INPUT_W, INPUT_H, inverse=True)
    crops = [cv2.warpAffine(f, matrix, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR) for f in frames]
    coords, scores = decode_heatmaps(run_crops(runner, crops), STRIDE)
    coords = np.stack([apply_affine(c, inverse) for c in coords])
    return np.concatenate([coords, scores[..., None]], axis=-1).astype(np.float32)


def coco_scores(heatmaps, data):
    """Same metric as models/keypoints/train_keypoints.py: error / longer side of the person box."""
    coords, scores = decode_heatmaps(heatmaps, STRIDE)
    coords = np.stack([apply_affine(c, m) for c, m in zip(coords, data["inverse"])])
    gt = data["kpts"]
    errors = np.linalg.norm(coords - gt[:, :, :2], axis=-1) / data["box_side"][:, None]
    labeled = gt[:, :, 2] > 0
    metrics = {}
    for group, group_slice in list(GROUP_SLICES.items()) + [("all", slice(None))]:
        err = errors[:, group_slice][labeled[:, group_slice]]
        metrics[f"{group}_nme"] = float(err.mean())
        for threshold in (0.05, 0.1):
            metrics[f"{group}_pck@{threshold}"] = float((err < threshold).mean())
    keypoints = np.concatenate([coords, scores[..., None]], axis=-1).astype(np.float32)
    return metrics, keypoints


def print_metrics(metrics):
    print(f"  {'group':12s} {'PCK@0.05':>9s} {'PCK@0.1':>9s} {'NME':>7s}")
    for group in list(GROUP_SLICES) + ["all"]:
        print(f"  {group:12s} {metrics[f'{group}_pck@0.05'] * 100:8.1f}% {metrics[f'{group}_pck@0.1'] * 100:8.1f}% "
              f"{metrics[f'{group}_nme']:7.4f}")


def step_check(runner, args):
    data = np.load(os.path.join(args.data_dir, "coco_eval.npz"))
    crops = data["crops"][:32]
    sample = {k: data[k][:32] for k in ("inverse", "kpts", "box_side")}
    raw = unwrap(runner.infer(np.ascontiguousarray(crops[0].astype(np.float32) / 255.0)))
    print(f"[check] accelerator output shape {raw.shape}, expected {runner.out_chw} (CHW) or its HWC transpose")
    candidates = [runner.layout] if runner.layout != "auto" else ["chw", "hwc"]
    results = {}
    for layout in candidates:
        try:
            results[layout] = coco_scores(run_crops(runner, crops, layout=layout), sample)[0]["all_pck@0.1"]
        except AssertionError as error:
            print(f"[check] layout {layout}: {error}")
    for layout, pck in results.items():
        print(f"[check] layout {layout}: PCK@0.1 on 32 COCO people = {pck * 100:.1f}%")
    runner.layout = max(results, key=results.get)
    print(f"[check] using layout {runner.layout}. FP32 on these 32 people scores about 95%; "
          "far below that means something is wrong, so stop and send back the output above.")
    return {"layout": runner.layout, "check_pck@0.1": results[runner.layout]}


def step_coco(runner, args):
    data = np.load(os.path.join(args.data_dir, "coco_eval.npz"))
    print(f"[coco] {len(data['crops'])} people")
    start = time.time()
    metrics, keypoints = coco_scores(run_crops(runner, data["crops"], log_every=500), data)
    metrics["seconds"] = round(time.time() - start, 1)
    print_metrics(metrics)
    np.save(os.path.join(args.results, "coco_keypoints.npy"), keypoints)
    with open(os.path.join(args.results, "coco_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return {"coco_all_pck@0.1": metrics["all_pck@0.1"]}


def step_speed(runner, args, iterations=300, warmup=20):
    crop = np.load(os.path.join(args.data_dir, "coco_eval.npz"))["crops"][0].astype(np.float32) / 255.0
    for _ in range(warmup):
        runner.infer(crop)
    start = time.perf_counter()
    for _ in range(iterations):
        runner.infer(crop)
    accel_ms = (time.perf_counter() - start) / iterations * 1000

    clips = read_clips(args)
    frames = np.load(os.path.join(args.data_dir, "wlasl", "frames", f"{clips[0]['video_id']}.npy"))
    predict_frames(runner, frames)
    start = time.perf_counter()
    reps = 10
    for _ in range(reps):
        predict_frames(runner, frames)
    end_to_end_ms = (time.perf_counter() - start) / (reps * len(frames)) * 1000

    speed = {
        "accelerator_ms_per_frame": round(accel_ms, 3), "accelerator_fps": round(1000 / accel_ms, 1),
        "end_to_end_ms_per_frame": round(end_to_end_ms, 3), "end_to_end_fps": round(1000 / end_to_end_ms, 1),
        "note": "accelerator = one infer() call; end to end adds crop, layout fix, plan-B head if used, heatmap decoding",
    }
    print(f"[speed] accelerator {speed['accelerator_ms_per_frame']} ms/frame ({speed['accelerator_fps']} fps); "
          f"end to end {speed['end_to_end_ms_per_frame']} ms/frame ({speed['end_to_end_fps']} fps)")
    with open(os.path.join(args.results, "speed.json"), "w", encoding="utf-8") as f:
        json.dump(speed, f, indent=2)
    return speed


def step_domain_gap(runner, args):
    label_set = np.load(os.path.join(args.data_dir, "label_set.npz"))
    keypoints = predict_frames(runner, label_set["frames"])
    np.save(os.path.join(args.results, "domain_gap_keypoints.npy"), keypoints)
    print(f"[domain_gap] keypoints for {len(keypoints)} frames saved (scored on the laptop)")
    return {}


def read_clips(args):
    with open(os.path.join(args.data_dir, "wlasl", "clips.csv"), "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def step_wlasl(runner, args):
    clips = read_clips(args)
    for condition in CONDITIONS:
        out_dir = os.path.join(args.results, "wlasl", condition)
        os.makedirs(out_dir, exist_ok=True)
        start, done = time.time(), 0
        for clip in clips:
            vid = clip["video_id"]
            out_path = os.path.join(out_dir, f"{vid}.npy")
            if os.path.isfile(out_path):
                continue
            frames = np.load(os.path.join(args.data_dir, "wlasl", "frames", f"{vid}.npy"))
            if condition != "clean":
                degradation, severity = condition.split("_")
                hand = np.load(os.path.join(args.data_dir, "wlasl", "hand_keypoints", f"{vid}.npy"))
                frames, _ = degrade_clip(frames, degradation, severity, clip_rng(vid, degradation, severity, 0), hand)
            np.save(out_path, predict_frames(runner, frames))
            done += 1
        print(f"[wlasl] {condition:20s} {done:4d} new clips ({time.time() - start:.0f}s)", flush=True)
    return {}


STEPS = {"check": step_check, "coco": step_coco, "speed": step_speed, "domain_gap": step_domain_gap, "wlasl": step_wlasl}


def main():
    ap = argparse.ArgumentParser(description="Run the Stage 1 keypoint extractor on the Mobilint NPU.")
    ap.add_argument("step", choices=list(STEPS) + ["all"])
    ap.add_argument("--backend", choices=["npu", "onnx"], default="npu")
    ap.add_argument("--mxq", default=os.path.join(HERE, "model.mxq"))
    ap.add_argument("--onnx", default=os.path.join(HERE, "model.onnx"), help="Model for --backend onnx")
    ap.add_argument("--head_onnx", help="Plan B: run this deconv head on the CPU after the backbone .mxq")
    ap.add_argument("--layout", choices=["auto", "chw", "hwc"], default="auto",
                    help="Layout of the accelerator output; 'auto' is settled by the check step")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--results", help="Default: results_<backend>[_split] next to this script")
    args = ap.parse_args()
    args.results = args.results or os.path.join(HERE, f"results_{args.backend}" + ("_split" if args.head_onnx else ""))
    os.makedirs(args.results, exist_ok=True)

    runner = Runner(args)
    info_path = os.path.join(args.results, "run_info.json")
    info = json.load(open(info_path, encoding="utf-8")) if os.path.isfile(info_path) else {}
    info.update({"backend": args.backend, "model": args.mxq if args.backend == "npu" else args.onnx,
                 "head_onnx": args.head_onnx, "python": platform.python_version(), "machine": platform.node()})
    if args.layout == "auto" and info.get("layout"):
        runner.layout = info["layout"]

    steps = ["check", "coco", "speed", "domain_gap", "wlasl"] if args.step == "all" else [args.step]
    if runner.layout == "auto" and "check" not in steps:
        steps = ["check"] + steps
    for step in steps:
        info.update(STEPS[step](runner, args))
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
    print(f"Done. Results in {args.results}")


if __name__ == "__main__":
    main()
