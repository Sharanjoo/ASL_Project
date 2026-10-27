# ASL Sign Recognition — Data Setup & Preprocessing

This kit covers the data setup and preprocessing milestone of the ASL sign-language
keypoint recognition project: a CNN-based keypoint extraction pipeline trained on
WLASL signing video, with COCO-WholeBody used as keypoint supervision data for the
upstream keypoint model.

## Project layout

```
asl_project/
  data/
    wlasl/
      select_subset.py        # rank + filter WLASL glosses into a class subset
      sample_frames.py        # sample multi-frame numpy arrays from video clips
    coco_wholebody/
      explore_annotations.py  # sanity-check keypoint annotations against images
    sample_out/                # demo/verification outputs land here
  raw_videos/                   # downloaded WLASL clips go here
  subset_100/                   # select_subset.py / sample_frames.py outputs go here
  requirements.txt
  README.md
```

## Setup

```
pip install -r requirements.txt
```

## Getting WLASL video data

Clone the WLASL repo to get the metadata file and the video downloader:

```
git clone https://github.com/dxli94/WLASL.git
```

This gives you `WLASL_v0.3.json` and `start_kit/video_downloader.py`. The downloader
pulls clips from three sources depending on what each instance's metadata points to:
YouTube (via `yt-dlp`), ASLPro (a custom `.swf` download), and a handful of other
hosts served as plain HTTP `.mp4`. Point its output at `raw_videos/`.

**A large number of the YouTube links will be dead.** The metadata dates back to
2020, and a meaningful fraction of the referenced videos have since been removed,
made private, or region-locked. This is expected, not a bug in the downloader —
`sample_frames.py` below counts and reports these as "missing video" rather than
failing on them.

As a more reliable alternative (worth starting in parallel, since response time
varies), the WLASL team also offers a pre-processed-video request form:
https://docs.google.com/forms/d/e/1FAIpQLSc3yHyAranhpkC9ur_Z-Gu5gS5M0WnKtHV07Vo6eL6nZHzruw/viewform?usp=sf_link

## Selecting a class subset

```
python data/wlasl/select_subset.py \
    --wlasl_json WLASL_v0.3.json \
    --out_dir subset_100 \
    --num_classes 100 \
    --min_instances_per_class 1
```

Classes are ranked by instance count, not alphabetically, and the top
`--num_classes` are kept. WLASL has a long tail of glosses with only one or two
clips; selecting by count instead of alphabetically avoids starving a chosen class
of training data. The script writes `manifest.csv` (one row per instance) and
`label_ids.csv` (the gloss → label_id mapping) into `--out_dir`, and prints a
summary of classes selected, total instances, and per-split counts.

## Sampling frames from video clips

```
python data/wlasl/sample_frames.py \
    --manifest subset_100/manifest.csv \
    --video_dir raw_videos \
    --out_dir subset_100/frames \
    --num_frames 5 \
    --img_size 224
```

This samples several evenly-spaced frames per clip rather than a single frame.
Single-frame sampling discards the motion information that distinguishes one sign
from another — published WLASL100 baselines show a large accuracy jump from adding
temporal modeling (roughly 26% top-1 for a frame-only VGG-GRU model versus roughly
66% top-1 for a full spatiotemporal I3D model), so multi-frame sampling here is a
deliberate design choice, not an arbitrary default.

Check the OK / missing-video / bad-clip counts the script prints at the end — a
high missing-video count is expected given the dead-link issue above, but a high
bad-clip count can indicate a video format OpenCV is failing to decode and is worth
investigating.

## Getting COCO-WholeBody data

Images are not redistributed by COCO-WholeBody — download the standard COCO 2017
`train2017`/`val2017` image sets from https://cocodataset.org/#download. Annotations
come separately from https://github.com/jin-s13/COCO-WholeBody (OneDrive, GDrive,
BaiduPan, and OpenXLab links are listed in their README).

Each annotation record covers 133 keypoints across five groups:

| Field             | Points | Array length |
|-------------------|--------|--------------|
| `keypoints`       | 17     | 51           |
| `foot_kpts`       | 6      | 18           |
| `face_kpts`       | 68     | 204          |
| `lefthand_kpts`   | 21     | 63           |
| `righthand_kpts`  | 21     | 63           |

Each point is an `(x, y, v)` triple. `v` is a visibility flag: `0` = unlabeled,
`1` = labeled but not visible, `2` = labeled and visible. Points with `v == 0` must
be skipped when drawing or training — they carry no real coordinate information.

## Verifying the annotation parsing

Run in demo mode first — this builds a synthetic image and a fake annotation using
the real field names, so you can confirm the parsing and drawing logic before
downloading any real data:

```
python data/coco_wholebody/explore_annotations.py --demo --out data/sample_out/demo_overlay.png
```

Then run against real data:

```
python data/coco_wholebody/explore_annotations.py \
    --ann coco_wholebody_train_v1.0.json \
    --img_dir train2017 \
    --out data/sample_out/real_overlay.png
```

A correct overlay has the keypoints and boxes landing on the actual person in the
image — face points on the face, hand points on the hands, and so on. If the points
are scattered across the image with no relation to the subject, the most common
cause is pairing the annotation file with the wrong image split (e.g. a `val2017`
annotation file matched against the `train2017` image folder).

## Things that will probably go wrong and aren't bugs

- **Missing WLASL videos** from dead YouTube links — the metadata is from 2020 and
  link rot is expected. Use the pre-processed-video request form in parallel if
  this is blocking progress.
- **`.mkv`/`.webm` codec issues** after `yt-dlp` remuxes a download — some codecs
  aren't readable by every OpenCV build. If `sample_frames.py` reports an unusually
  high bad-clip count, check which extensions those clips have.
- **COCO-WholeBody image_id/split mismatches** — pairing the wrong `train2017`/
  `val2017` folder with an annotation file produces scattered, nonsensical overlays
  rather than an error, so this is worth checking visually with
  `explore_annotations.py` before trusting the data.

## Stage 1 — spatial keypoint extractor (weeks 3–4)

`models/keypoints/` holds the heatmap-based CNN keypoint extractor from the proposal:
RGB person crop → ResNet backbone → 3 deconv layers → one heatmap per keypoint
(stride 4) → (x, y) coordinates. It predicts 73 signing-relevant points:

| Group       | Points | Source in COCO-WholeBody                                      |
|-------------|--------|---------------------------------------------------------------|
| upper body  | 11     | body 0–10 (nose, eyes, ears, shoulders, elbows, wrists)       |
| mouth       | 20     | face 48–67 (iBUG outer + inner lip)                           |
| left hand   | 21     | `lefthand_kpts`                                               |
| right hand  | 21     | `righthand_kpts`                                              |

Files:

```
models/keypoints/
  keypoint_spec.py            # 73-point subset, groups, left/right flip pairs
  transforms.py               # crop affine, Gaussian heatmap targets, heatmap decoding
  coco_wholebody_dataset.py   # top-down person-crop dataset with scale/rotate/flip augmentation
  model.py                    # ResNet + deconv heatmap network, weighted MSE loss
  train_keypoints.py          # training loop, per-group PCK / NME evaluation, checkpoints
  predict_wlasl.py            # run a checkpoint over manifest_frames.csv clips
  setup_gpu_data.sh           # download COCO 2017 images + WholeBody annotations
  train_gpu.sh                # full training command (ResNet-50, 256x192, 140 epochs)
```

Evaluation reports, per group and overall, PCK@0.05 / PCK@0.1 and mean normalized
error, where a prediction's pixel error is divided by the longer side of the person
bounding box. Only labeled points (`v > 0`) are scored.

### Full training on a GPU machine

Copy the `asl_project` folder (code only — the data folders are re-downloaded there),
then from inside it:

```
bash models/keypoints/setup_gpu_data.sh
bash models/keypoints/train_gpu.sh
```

`setup_gpu_data.sh` needs about 20 GB and skips anything already downloaded.
`train_gpu.sh` forwards extra flags to `train_keypoints.py`, so a run that hits a job
time limit continues with:

```
bash models/keypoints/train_gpu.sh --resume runs/keypoints_resnet50_256x192/last.pt
```

Checkpoints (`best.pt`, `last.pt`), `train_log.csv`, and `final_metrics.json` land in
the run's `--out_dir`.

### CPU smoke test

Without a GPU, a short run on val2017 alone checks the whole pipeline. With no
`--val_ann`, every 10th `image_id` is held out for evaluation:

```
python models/keypoints/train_keypoints.py \
    --train_ann coco_wholebody_ann/coco_wholebody_val_v1.0.json \
    --train_img_dir coco_images/val2017 \
    --out_dir runs/keypoints_smoke \
    --input_h 128 --input_w 96 --sigma 1.5 --epochs 10
```

This only confirms the code trains end to end; its accuracy is not a Stage 1 result.

### Running the extractor on WLASL frames

```
python models/keypoints/predict_wlasl.py \
    --checkpoint runs/keypoints_resnet50_256x192/best.pt \
    --manifest subset_100/frames/manifest_frames.csv \
    --out_dir subset_100/keypoints
```

Each clip becomes a `(num_frames, 73, 3)` array of `(x, y, score)` in the 224×224
frame's pixel coordinates, listed in `manifest_keypoints.csv`. The first few clips
also get an overlay PNG — use these for the proposal's domain-gap check, since the
extractor is trained on natural COCO images, not signing footage.

`sample_frames.py --crop bbox` (the default) takes a square crop around each clip's
WLASL signer bbox, padding with black where the square leaves the frame, so the
signer is centered and undistorted. The bbox is stored at the source video's
resolution; clips downloaded smaller (e.g. 360p YouTube against 720p metadata) are
rescaled by how far the box overshoots the frame. `--crop letterbox` pads the whole
frame to a square instead, and `--crop stretch` reproduces the original
resize-straight-to-square behavior.

The later stages use 16 frames per clip:

```
python data/wlasl/sample_frames.py \
    --manifest subset_100/manifest.csv --video_dir raw_videos \
    --out_dir subset_100/frames16 --num_frames 16 --img_size 224
```

### Domain-gap check (hand-labeled WLASL frames)

The extractor learns from COCO photos, so the proposal checks it on signing video
directly. `models/keypoints/domain_gap/` builds a 100-frame label set from the test
split (middle frame of each clip, round-robin across signers), with 19 points per frame:
nose, shoulders, elbows, wrists, 4 mouth points, and 4 fingertips per hand.

```
python models/keypoints/domain_gap/make_label_set.py \
    --manifest subset_100/frames16/manifest_frames.csv --out_dir domain_gap
```

Open `domain_gap/label_tool.html` in a browser (it is self-contained; nothing is
uploaded). Click each highlighted point, press N for points that are hidden, Z to undo.
Labels autosave in the browser; **Export** writes a JSON file. Several people can split
the frames and their exports are merged at scoring time:

```
python models/keypoints/domain_gap/score_domain_gap.py \
    --checkpoint runs/keypoints_resnet50_256x192/best.pt \
    --label_set domain_gap/label_set.npz --labels domain_gap_labels_*.json
```

It reports PCK@0.1/0.2/0.5 and mean error per group, as a fraction of the labeled
shoulder width (frames without both shoulders are skipped). Each wrist label is scored
against both the body wrist and the hand-root keypoint.

## Report figures

`python report/make_figures.py` writes the dataset coverage chart, the degradation
examples and the 73-keypoint layout to `report/figures/`. After the GPU run, add
`--hand_keypoints_dir runs/pipeline/keypoints/clean` so the occlusion examples sit on
the wrists.

## Controlled degradation (week 9)

`data/degradation/degrade.py` implements the proposal's three degradations at three
severities. Each is applied on the fly inside `predict_wlasl.py` (`--degradation`,
`--severity`), so only keypoints are saved, never degraded frame copies.

| Severity | Frames hit | Occluder side | Blur kernel |
|----------|------------|---------------|-------------|
| mild     | 25%        | 20% of frame  | 7 px        |
| moderate | 50%        | 25% of frame  | 13 px       |
| severe   | 75%        | 30% of frame  | 21 px       |

- **occlusion** — a gray square over one hand (same hand for the whole clip),
  placed on that wrist from the clean run's keypoints (`--hand_keypoints_dir`) when
  its score is ≥ 0.3, otherwise at a typical signing-hand position.
- **blur** — linear motion blur at a random angle per frame.
- **missing** — the hit frames are blacked out.

At least one frame always stays clean. The randomness is seeded per
video/degradation/severity, so every run degrades the same frames. Preview all
nine settings on one clip with:

```
python data/degradation/degrade.py --frames subset_100/frames16/<video_id>.npy --out data/sample_out/degradations.png
```

## Stage 2 — temporal keypoint refinement

`models/refinement/` refines frame t's keypoints from a t-2…t+2 window (Transformer
by default, `--arch gru` available). It predicts a coordinate correction plus a new
confidence score per point. The target is always the extractor's output on the
**clean** frame. Target points with score ≥ 0.2 get full weight in the loss and the
rest get weight 0.1, so low-confidence points stay anchored instead of drifting
(with zero weight they drifted ~75 px and wiped out the classifier's input). Two
training modes match the proposal's ablation:

- `--mode synthetic` — inputs are clean keypoints with random point/frame dropout
  and jitter (Baseline B: temporal refinement, no robustness training).
- `--mode degraded` — inputs are the extractor's keypoints on visually degraded
  frames (Proposed: temporal refinement + robustness training).

`refine_keypoints.py` applies a trained refiner to any keypoint directory.

## Stage 3 — sign recognition

`models/recognition/` classifies a clip's (T, 73, 3) keypoint sequence into one of
the 100 glosses (bidirectional GRU by default, `--arch transformer` available). It
reports top-1, top-5 and macro F1 on val/test, keeping the checkpoint with the best
val top-1.

## Running everything after Stage 1

```
bash models/run_pipeline.sh <keypoint checkpoint> runs/pipeline
```

This extracts keypoints for clean frames and all 9 degraded conditions, trains both
refiners and the three classifiers (Frame-only, +Temporal, +Temporal & robust
training), then fills the proposal's robustness table: `robustness.md` and
`robustness.csv` in the output folder, test split. With the CPU smoke-test
checkpoint it only checks the wiring; meaningful numbers need the GPU-trained
extractor.
