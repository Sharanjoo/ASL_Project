# Handoff for the RTX 4090 machine (read this first, then README.md and GPU_RUN.md)

## Project in one paragraph
DATA 255, SJSU, Group 2. ASL word recognition (WLASL100) from keypoints, tested under occlusion,
motion blur and missing frames. Three stages, each trained and saved on its own:
1. **Stage 1** keypoint extractor: ResNet-50 + 3 deconv layers, 73 heatmaps, trained on COCO-WholeBody (`models/keypoints/`).
2. **Stage 2** temporal refiner: Transformer over the clip, repairs keypoints with score < 0.2 (`models/refinement/`).
3. **Stage 3** classifier: 2-layer bi-GRU over 16 frames x 73 keypoints -> 100 glosses (`models/recognition/`).

Robustness table (`models/evaluate_robustness.py`) columns: Frame-only, +Interpolation (Roh et al. 2024,
no training), +Temporal (refiner trained on synthetic noise), Proposed (refiner trained on real degraded frames).
10 conditions: clean + {occlusion, blur, missing} x {mild, moderate, severe}. Metrics: top-1, top-5, macro-F1 on the
161 test clips. Full details: `README.md`. GPU walkthrough: `GPU_RUN.md`.

## Where things stand
- Stage 1 trained earlier on a different GPU machine: 95.2% PCK@0.1 on COCO val (93.4% hands), 98.2% PCK@0.2 on our 100 hand-labeled WLASL frames.
- Reported results (top-1, 161 test clips, one run per model, no significance test): average over the 9 degraded conditions
  Frame-only 22.3, +Interpolation 24.9, +Temporal 20.9, Proposed 28.0. Clean: 30.4 / 32.3 / 26.7 / 32.3.
- Mobilint NPU path exists (`models/npu/`); NPU retrain of Stages 2-3 is waiting on a lab run. Not a 4090 task.
- Only **1,020 of 2,038** WLASL100 clips are available (dead links; `missing.txt` lists 1,017 missing IDs).
  Approval from the WLASL authors for the rest is pending; a public Kaggle copy is being considered.
  Verify a Kaggle copy against `subset_100/manifest.csv` and `WLASL_v0.3.json` (same clip IDs, official splits) before using it,
  and keep the 100 domain-gap test clips in the test split. WLASL is C-UDA licensed: keep the clips private.

## NOT in git, must be copied over (gitignored)
- `raw_videos/` and `subset_100/` (manifest, `frames16/`, keypoints), about 540 MB zipped (`asl_wlasl_videos.zip`).
- `runs/keypoints_resnet50_256x192/best.pt` (Stage 1 checkpoint) and `runs/pipeline/` (refiners, classifiers, `keypoints/`).
- COCO 2017 + WholeBody annotations are re-downloaded by `bash models/keypoints/setup_gpu_data.sh` (~20 GB) and only needed to retrain Stage 1.
- `domain_gap/label_set.npz` and the label tool HTML are gitignored (rebuild with `make_label_set.py`); the exported labels JSON is in git.

## Uncommitted work on the original machine (copy it or commit and push first)
`models/raw_frame/` (new), plus edits to `models/run_pipeline.sh` and `README.md`. The raw-frame baseline:
ImageNet ResNet-18 on every frame -> same bi-GRU head as Stage 3, trained on clean frames, evaluated on all 10 conditions
with the same seeded degradation (`clip_rng`) so it sees the identical damage. It was only smoke-tested on synthetic
clips, never on real data.

## Setup on the 4090 machine
```bash
python -m venv .venv && source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128   # the 40-series needs a CUDA 12 build
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"                       # must print True
```
`run_all_gpu.sh` and the `.sh` scripts need bash (Linux, WSL or Git Bash). Use relative paths in `--config` strings:
`evaluate_robustness.py` splits on `:`, so an absolute `C:\...` path breaks it.

## Jobs for this machine, in priority order
1. **Raw-frame CNN baseline** (the professor-facing comparison: do pixels degrade more than keypoints?).
   ```bash
   python models/raw_frame/train_raw_frames.py --manifest subset_100/frames16/manifest_frames.csv \
       --out_dir runs/pipeline/classifier_raw_frame --batch_size 16
   python models/raw_frame/evaluate_raw_frames.py --checkpoint runs/pipeline/classifier_raw_frame/best.pt \
       --manifest subset_100/frames16/manifest_frames.csv --hand_keypoints_dir runs/pipeline/keypoints/clean \
       --out runs/pipeline/robustness_raw_frame.csv --merge_into runs/pipeline/robustness.csv
   ```
   The first run downloads ImageNet weights (needs internet). 24 GB allows larger batches than the defaults.
   Expect low absolute accuracy (published VGG-GRU is ~26% top-1 on the full WLASL100); this is a robustness comparison,
   not an accuracy race. Try `--freeze_backbone` if the fine-tuned run overfits 680 training clips.
2. **MediaPipe and OpenPose/RTMPose baselines (NOT BUILT YET)**: the professor asked for these. Nothing in the repo does this.
   Plan: (a) score them on the 100 hand-labeled WLASL frames with `models/keypoints/domain_gap/score_domain_gap.py`'s
   protocol (PCK as a fraction of shoulder width, 19 points); (b) on COCO-WholeBody val for the shared points;
   (c) extract their keypoints for all 1,020 clips clean + 9 degradations, map to the 73-point layout
   (`models/keypoints/keypoint_spec.py`), score 0 for undetected points, and feed the same Stage 3 recipe so they
   become extra columns in the robustness table. MediaPipe runs on CPU; RTMPose/RTMW (MMPose) is the practical stand-in
   for OpenPose, which is hard to install. Put scripts in a new `models/baselines/` folder.
3. **Seeds and significance**: retrain Stages 2-3 (and the raw-frame model) with several `--seed` values and report mean +/- std.
   Current margins are about 5 clips per condition, so this is the main weakness of the results.
4. **Rerun everything if more WLASL clips arrive** (clear `subset_100/frames16`, `runs/pipeline`, then `bash run_all_gpu.sh`;
   Stage 1 is unaffected because it trains on COCO). Old and new numbers are not comparable.
5. Only if needed: Stage 1 retraining via `bash models/keypoints/train_gpu.sh` (batch 64 should fit in 24 GB).

## Gotchas
- Quick run flags: `REFINER_EPOCHS=10 CLASSIFIER_EPOCHS=60 bash models/run_pipeline.sh <ckpt> <out>`; `SKIP_KEYPOINTS=1` reuses extracted keypoints.
- `RAW_FRAME=1 bash models/run_pipeline.sh ...` runs the raw-frame baseline as a last step.
- The refiner loss gives low-confidence target points weight 0.1, not 0 (weight 0 made points drift ~75 px).
- Blur barely helps interpolation because blurred keypoints keep high confidence and are never replaced; occlusion results are within ~8 clips for all configs, so do not claim a win there.
- Do not fabricate numbers: everything in the report and slides comes from `runs/pipeline/robustness.csv`.
- Do not commit or push without the user asking. Do not publish WLASL clips or frames.
