# ASL project: Stage 1 keypoint extractor on the Mobilint NPU

This bundle compiles our keypoint model to INT8 with Qubee and runs it on the ARIES2 NPU with MACCEL. It
uses the same two tools and the same machine as DATA 255 Labs 1 and 2. You don't need PyTorch, GPUs, COCO
or the WLASL videos: everything the NPU needs is in this folder.

**Time:** about 1 to 1.5 hours, mostly waiting. **Send back:** one small zip, about 20 MB (step 4).

## What the model is

- **Model:** ResNet-50 plus 3 upsampling (ConvTranspose) layers, trained on COCO-WholeBody.
- **Input:** a 256 x 192 RGB crop of a person, as `(256, 192, 3)` float32 values in [0, 1]. This is the
  same layout as Lab 2. The ImageNet normalisation is inside the model.
- **Output:** 73 heatmaps of 64 x 48, one per keypoint (body, mouth, both hands).
- **ONNX ops:** Conv, ConvTranspose, Relu, Add, Mul, MaxPool and one input Transpose. BatchNorm is
  already folded into the weights.

## What's in the folder

| Item | What it is |
| --- | --- |
| `model.onnx` | The full model (plan A) |
| `model_backbone.onnx`, `model_head.onnx` | Plan B, only if Qubee can't handle ConvTranspose (step 3b) |
| `calib_data/` | 300 calibration crops, already at 256 x 192: 150 COCO people plus 150 WLASL training frames. None are used for evaluation. |
| `convert_to_mxq.py` | Qubee compile script (runs in Docker). Same settings as the labs: per-channel quantization, max-percentile, 0.9999. |
| `run_npu.py` | Runs everything on the NPU with MACCEL |
| `data/` | Evaluation inputs: 919 COCO people, 100 hand-labeled WLASL frames and 161 WLASL test clips |
| `lib/` | Helper code `run_npu.py` imports (cropping, heatmap decoding, degradations) |

## Step 1: Put the folder in place

Unzip so you have `C:\NPU_Projects\asl_npu_bundle\run_npu.py`, next to the Qubee wheel used in the labs
(`C:\NPU_Projects\qubee-0.12.0.0+aries2-py3-none-any.whl`).

## Step 2: Compile to .mxq in the Qubee Docker container

```bat
docker run -it --name qb_asl -v C:/NPU_Projects:/NPU_Projects mobilint/qbcompiler:0.12-cpu-ubuntu22.04 /bin/bash
```

If `qb_asl` already exists, run `docker start -ai qb_asl` instead. Then, inside the container:

```bash
pip install /NPU_Projects/qubee-0.12.0.0+aries2-py3-none-any.whl
pip install numpy pillow
cd /NPU_Projects/asl_npu_bundle
python convert_to_mxq.py --onnx model.onnx --out model.mxq 2>&1 | tee compile_full.log
```

Compiling takes a few minutes. Look for:

- **The line with the percentage of supported ops.** In the labs it said 97.5%. Note the number and
  which ops it lists as unsupported.
- **`[mxq] DONE -> model.mxq`.** If you see this, go to step 3a.
- **An error, or ConvTranspose listed as unsupported.** Go to step 3b.

## Step 3a: Run on the NPU (plan A)

On Windows, outside Docker, use **the same Python 3.12 you used for `eval_on_npu.py` in the labs** (the
one with the `maccel` wheel). Below it's written as `python312`; replace that with its full path.

```bat
cd C:\NPU_Projects\asl_npu_bundle
python312 -m pip install numpy opencv-python onnxruntime
python312 run_npu.py check
```

`check` runs 32 COCO people and prints a line like:

```
[check] layout chw: PCK@0.1 on 32 COCO people = 95.6%
```

- **Around 85% or higher:** fine. On the laptop in float (no quantization) it's 95.6%.
- **Far below that** (both layouts under about 50%): stop, and send back the whole output.

Then run the full job:

```bat
python312 run_npu.py all > run_npu.log 2>&1
```

This writes `results_npu\`. It runs, in order:

- `check`
- `coco`: 919 people, with a PCK table printed at the end
- `speed`: milliseconds per frame
- `domain_gap`: 100 frames
- `wlasl`: 161 clips x 10 conditions = 25,760 frames. This is the long part.

If it stops partway, just run the same command again: finished WLASL clips are skipped.

**What to compare against.** These are the same 919 people run in float (ONNX on CPU) on the laptop. The
NPU's INT8 table should land within a few points of it:

| Group | PCK@0.05 | PCK@0.1 |
| --- | --- | --- |
| upper_body | 88.7% | 95.0% |
| mouth | 98.7% | 99.0% |
| left_hand | 88.4% | 93.6% |
| right_hand | 88.2% | 93.8% |
| all | 90.9% | 95.2% |

For speed, the laptop CPU did 74 ms per frame (13.6 fps).

**Optional, recommended:** time the same model on this machine's CPU, for a fair speed comparison. It
only times the model, and takes a couple of minutes:

```bat
python312 run_npu.py speed --backend onnx
```

## Step 3b: Plan B, only if step 2 failed on ConvTranspose

Compile just the ResNet-50 backbone, which is most of the compute, and run the small head on the CPU.

In Docker:

```bash
python convert_to_mxq.py --onnx model_backbone.onnx --out model_backbone.mxq 2>&1 | tee compile_backbone.log
```

On Windows:

```bat
python312 run_npu.py all --mxq model_backbone.mxq --head_onnx model_head.onnx > run_npu_split.log 2>&1
```

This writes `results_npu_split\`.

## Step 4: Send these back

Zip and send:

- **`results_npu\`**, or `results_npu_split\` for plan B
- **`results_onnx\`**, if you ran the optional CPU timing
- **`compile_full.log`**, and `compile_backbone.log` for plan B
- **`run_npu.log`**, or `run_npu_split.log`

That's everything. Sharan scores it on the laptop.

## If something goes wrong

| Problem | What to do |
| --- | --- |
| `ModuleNotFoundError: cv2` / `onnxruntime` | `python312 -m pip install opencv-python onnxruntime` |
| `maccel` won't import | That Python isn't the lab's 3.12 with the maccel wheel. Use the one from Lab 1/2. |
| `check` prints an output size error | Send `run_npu.log`. The compiled model's output isn't 73 x 64 x 48. |
| `check` is far below 85% | Send the output. Try `--layout chw` and `--layout hwc` by hand. |
| The WLASL step is very slow | Leave it running. It prints one line per condition (10 total). |
