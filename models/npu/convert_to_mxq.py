r"""convert_to_mxq.py -- Stage 1 keypoint extractor: ONNX -> Mobilint .mxq (INT8) with Qubee 0.12.
Run inside the Qubee Docker container (see README_NPU.md, step 2):

    python /NPU_Projects/asl_npu_bundle/convert_to_mxq.py --onnx model.onnx --out model.mxq

Calibration images are already cropped to the model input (256 x 192) by make_npu_bundle.py: 150 COCO
people from images outside the COCO evaluation subset and 150 frames from WLASL train-split clips.
They are preprocessed here exactly as run_npu.py preprocesses its inputs: RGB, HWC, divided by 255.
The ImageNet normalisation is inside the ONNX graph.
"""
import argparse
import os

import numpy as np
from PIL import Image

INPUT_H, INPUT_W = 256, 192


def preprocess(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    assert img.size == (INPUT_W, INPUT_H), f"{img_path} is {img.size}, expected {(INPUT_W, INPUT_H)}"
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(here, "model.onnx"))
    ap.add_argument("--calib_dir", default=os.path.join(here, "calib_data"))
    ap.add_argument("--calib_save_dir", default=os.path.join(here, "calib_processed"))
    ap.add_argument("--out", default=os.path.join(here, "model.mxq"))
    ap.add_argument("--max_calib", type=int, default=300)
    ap.add_argument("--quantization_method", type=int, default=1)   # per-channel, as in the labs
    ap.add_argument("--quantization_mode", type=int, default=2)     # max-percentile, as in the labs
    ap.add_argument("--percentile", type=float, default=0.9999)     # gentle clipping: heatmap peaks must survive
    ap.add_argument("--quantization_output", type=int, default=0)
    args = ap.parse_args()

    files = [f for f in os.listdir(args.calib_dir) if f.lower().endswith(".png")]
    print(f"[mxq] {len(files)} calibration images in {args.calib_dir}")
    assert files, "No calibration images found."

    from qubee import mxq_compile
    from qubee.calibration import make_calib_man

    name = os.path.splitext(os.path.basename(args.onnx))[0] + "_cali"
    os.makedirs(args.calib_save_dir, exist_ok=True)
    make_calib_man(pre_ftn=preprocess, data_dir=args.calib_dir, save_dir=args.calib_save_dir,
                   save_name=name, max_size=args.max_calib)

    print(f"[mxq] compiling {args.onnx} -> {args.out}")
    mxq_compile(model=args.onnx,
                calib_data_path=os.path.join(args.calib_save_dir, name),
                quantization_method=args.quantization_method,
                quantization_mode=args.quantization_mode,
                percentile=args.percentile,
                quantization_output=args.quantization_output,
                save_path=args.out,
                backend="onnx")
    print(f"[mxq] DONE -> {args.out}")


if __name__ == "__main__":
    main()
