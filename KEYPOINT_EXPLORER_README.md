# ASL Keypoint Explorer — DATA 255 Group 2

A **Streamlit web application** for interactive COCO-WholeBody landmark selection, **Stage 1 ASL keypoint detection**, and inspection of saved keypoint predictions. The application is part of the group's American Sign Language (ASL) word-recognition research project.

## 1. Project overview and current scope

The team's intended ASL pipeline has three stages:

1. **Stage 1 — Keypoint detection:** A trained ResNet-50-based detector predicts **73 ASL-relevant landmarks** from an image or individual video frame.
2. **Stage 2 — Temporal refinement:** A Transformer-based model is intended to refine keypoints across sequences of video frames.
3. **Stage 3 — ASL word recognition:** A bidirectional GRU (BiGRU) is intended to classify ASL words from keypoint sequences.

**Current integration:** This Streamlit application **does run Stage 1 inference** on uploaded images and user-selected video frames when the trained checkpoint is available locally. It does **not** execute Stage 2 or Stage 3, recognize ASL words, or automatically process an entire video into a predicted sequence.

The app has three tabs:

- **Keypoint Selector:** Explore and select any of the 133 COCO-WholeBody landmarks; export selected IDs as JSON.
- **Image & Video:** Upload media, run trained Stage 1 detection on an image or selected video frame, inspect overlays, and download prediction arrays. Video frames can also be downloaded as original-resolution PNG files.
- **Model Predictions:** Load saved `.npy` predictions, optionally add a matching frame image, and inspect/filter overlays without rerunning the detector.

**Important distinction:** The 133-point reference skeleton in Tab 1 is illustrative, not output from the model. The trained detector currently predicts only 73 of the 133 landmarks. Changing landmark selection controls **what is visualized and exported**, not which outputs the model produces or how it is trained.

## 2. Requirements and installation

### Repository and files

Clone the team's repository:

```bash
git clone https://github.com/Sharanjoo/ASL_Project.git
cd ASL_Project
```

The app requires repository modules including:

- `keypoint_explorer.py` — Streamlit entry point.
- `models/keypoints/keypoint_spec.py` — `WHOLEBODY_INDICES` mapping.
- `models/keypoints/predict_wlasl.py` — Stage 1 model loading and prediction functions.

For **Stage 1 inference**, place the trusted trained PyTorch checkpoint at:

```text
ASL_Project/
├── keypoint_explorer.py
├── models/
│   └── keypoints/
│       ├── keypoint_spec.py
│       └── predict_wlasl.py
└── runs/
    └── keypoints_resnet50_256x192/
        └── best.pt
```

The file must be named **`best.pt`**, not `best.py`. The checkpoint is **not included in the GitHub commit** and must be obtained separately from an authorized teammate or trusted project storage. Do not commit large model weights to the repository without coordinating with the team.

Without `best.pt`, the keypoint selector, media preview, and saved prediction viewer remain usable, but the Stage 1 inference buttons are unavailable. The app displays whether the checkpoint was found.

> **Checkpoint safety:** The team's model loader uses `torch.load(..., weights_only=False)`. Only load checkpoint files from a trusted teammate or source; pickle-based model loading can execute code.

### Windows / PowerShell

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install streamlit streamlit-image-coordinates matplotlib numpy pillow opencv-python torch torchvision
python -m streamlit run keypoint_explorer.py
```

If PowerShell blocks environment activation, use your normal configured Python environment, or run the virtual environment's Python directly rather than changing system-wide security settings.

### macOS / Apple Silicon (including M4 Pro)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install streamlit streamlit-image-coordinates matplotlib numpy pillow opencv-python torch torchvision
python -m streamlit run keypoint_explorer.py
```

Tested during development with a MacBook Pro M4-series system using Python 3.14 and Apple Silicon package builds; other Python/package versions may require compatibility checks. Python 3.11 or 3.12 is a conservative alternative if dependency or checkpoint-loading issues arise.

**Device behavior:** The current `load_stage1()` implementation selects CUDA when available, otherwise **CPU**. On macOS it runs on CPU by default; it does **not** currently select Apple's MPS backend. CPU inference on an individual frame has been demonstrated on the MacBook.

### Launch / stop

Open the local URL shown by Streamlit, usually `http://localhost:8501`; if that port is in use, Streamlit may use `http://localhost:8502` or another port.

- Stop the server with **Ctrl+C** in the terminal.
- Relaunch with `python -m streamlit run keypoint_explorer.py` after activating `.venv`.
- Launch using **Streamlit**, not VS Code's plain **Run Python File** action.
- Once dependencies and the checkpoint are installed, local inference does not inherently require an internet connection.

### Dependency overview

| Package | Used for |
|---|---|
| `streamlit` | Web interface, uploads, interactive controls, session state |
| `streamlit-image-coordinates` | Clickable landmark visualization |
| `matplotlib` | Keypoint/connection plotting |
| `numpy` | Keypoint arrays and `.npy` serialization |
| `pillow` | Images and PNG frame export |
| `opencv-python` | Video decoding and selection of individual frames |
| `torch`, `torchvision` | Trained ResNet-50-based Stage 1 inference |

## 3. Application features

### Tab 1 — Keypoint Selector

Provides an **illustrative, interactive** 133-landmark COCO-WholeBody skeleton.

- **All 133:** Select every COCO-WholeBody ID (including 60 not supported by the current detector).
- **Current 73:** Restore the trained detector's supported landmark subset.
- **Clear:** Deselect everything.
- **Anatomical presets:** Add or remove face, mouth, upper body, torso/hips, legs, feet, left hand, and right hand groups.
- **Browse individual keypoints:** Filter to All 133 / Supported 73 / Future 60, then add or remove specific IDs.
- **Edit complete selected-ID list:** Use the multiselect for direct editing.
- **Interactive reference skeleton:** Click visible landmarks, toggle point IDs, show/hide unselected landmarks, and enable a larger view.
- **Selection summary:** See selected count, currently supported count, and selected-but-unsupported count.
- **Export selection (.json):** Save the selected original COCO-WholeBody IDs.

Color coding: **green** = body, **blue** = left hand, **orange** = right hand, **purple** = face/mouth.

The illustration is **not** a detection result. Selecting all 133 does not expand the model beyond its current 73 outputs. Selection state is held in the active Streamlit session; the app does not currently import exported selection JSON.

### Tab 2 — Image & Video (live Stage 1 inference)

When the checkpoint is present, the app displays **“Stage 1 checkpoint found locally.”**

**Image workflow**

1. Choose **Image** and upload PNG, JPG, or JPEG.
2. Review the uploaded image.
3. Click **Detect Stage 1 keypoints in image**.
4. Inspect the 73-point prediction overlay (subject to current selection and confidence settings).
5. Optionally adjust **Min confidence**, point IDs, visibility, or view size.
6. Click **Download image predictions (.npy)** to save a `(1, 73, 3)` array.

**Video workflow**

1. Choose **Video** and upload MP4, MOV, or AVI. Successful playback/decoding depends on installed codecs and browser support.
2. Use **Select frame** to choose one frame (slider is zero-based; display caption shows the one-based frame number).
3. Optionally click **Download original frame (.png)** to save the selected frame at its original pixel resolution.
4. Click **Detect Stage 1 keypoints in this frame**.
5. Inspect the overlay and click **Download frame predictions (.npy)** to save the matching `(1, 73, 3)` output.

For example, selecting the frame displayed as **Frame 32** produces `frame_32_original.png` and `frame_32_stage1.npy` when downloaded. Use matching filenames from the **same selected frame** when reviewing the result in Tab 3.

**Current behavior:** Video inference is **per selected frame**. The app does not automatically infer all frames, track across time, or export a full annotated video. A single-frame `.npy` export is not a full video sequence.

### Tab 3 — Model Predictions (saved results)

Tab 3 **does not execute the CNN**. It reads a saved `.npy` file containing either:

```python
(T, 73, 3)    # Stage 1 output, using the model's supported landmark order
(T, 133, 3)   # Full COCO-WholeBody indexed predictions
```

Each point contains `(x, y, confidence)`. `T` is the number of saved frames; **single-frame arrays (`T=1`) are supported** without a frame slider. Multi-frame arrays display a slider to choose the frame.

For 73-channel arrays, the viewer maps channels back to their original 133-landmark COCO indices using `WHOLEBODY_INDICES`.

**Workflow**

1. Upload the saved `.npy` file.
2. Optionally upload the **corresponding original frame PNG/image**.
3. Inspect the plotted landmark overlay, optionally toggle point IDs or unselected landmarks, and adjust **Min confidence**.
4. The confidence threshold is a **display filter only**; it does not change the saved predictions or model weights.

**Coordinate matching:** Predicted `(x, y)` positions must use the same pixel-coordinate system and frame dimensions as the background image. For video-frame results from Tab 2, use its **Download original frame (.png)** button instead of a screenshot or a resized image. A background image from a different frame, crop, or resolution may show misaligned or invisible keypoints. Without a background, the viewer automatically fits its axes around the available predicted coordinates.

## 4. COCO-WholeBody specification

| Region | COCO ID range | Total | Predicted by the current Stage 1 model |
|---|---|---:|---:|
| Body | `0–16` | 17 | 11 (`0–10`) |
| Feet | `17–22` | 6 | 0 |
| Face | `23–90` | 68 | 20 mouth points (`71–90`) |
| Left hand | `91–111` | 21 | 21 |
| Right hand | `112–132` | 21 | 21 |
| **Total** | **`0–132`** | **133** | **73** |

The remaining **60** COCO-WholeBody landmarks are available for selecting/planning but are not output by the trained detector.

## 5. Export formats

### Selection JSON (Tab 1)

Example file `keypoint_selection.json`:

```json
{
  "coco_wholebody_indices": [5, 6, 91, 112]
}
```

This file contains **selected IDs only**, not predictions, confidence scores, weights, or ASL classification results. JSON import is not currently implemented.

### Prediction NPY (Tab 2)

Prediction arrays have shape `(1, 73, 3)` for one image or one selected video frame. Each row is `(x, y, confidence)` in the model's 73-channel ordering. The viewer (Tab 3) performs the channel-to-COCO-ID mapping.

NPY files save the numerical predictions; they **do not include the underlying image**. Export the selected video's original frame PNG separately when a matching image is needed for visualization.

## 6. Functional test checklist

Before presenting or after changing the code, verify:

- [ ] **Keypoint Selector:** All 133 / Current 73 / Clear update selection counts.
- [ ] **Selection:** Anatomical presets, individual point editing, and clickable skeleton work.
- [ ] **JSON:** Export reflects the selected COCO IDs.
- [ ] **Checkpoint:** Tab 2 reports that `best.pt` is found.
- [ ] **Image inference:** Image upload → Stage 1 detection → visible landmark overlay.
- [ ] **Video inference:** Video upload → frame selection → Stage 1 detection → overlay.
- [ ] **Frame export:** Download original frame PNG and prediction NPY from the same video frame.
- [ ] **Saved viewer:** Import a `(1, 73, 3)` `.npy` into Tab 3, and verify a corresponding image overlay.
- [ ] **Confidence:** Adjusting the slider changes visibility of qualifying landmarks.
- [ ] **Mac demo:** Open and run the app on the actual presentation laptop, with local checkpoint and prepared demo media.

The image, video-frame, saved NPY, confidence, selection, and JSON interactions were manually exercised during development; rerun critical tests after future changes.

## 7. Current limitations

- **No Stage 2 integration:** The app does not perform Transformer-based temporal keypoint refinement.
- **No Stage 3 integration:** The app does not classify ASL words.
- **No automatic full-video inference:** Stage 1 runs only on the image or individual video frame chosen by the user.
- **Fixed 73-point detector:** Additional landmarks can be selected for visualization/planning, but the current model does not produce them.
- **Selection is not model reconfiguration:** Changing selected IDs does not alter training, architecture, learned parameters, or detector outputs.
- **Prediction-image alignment requires matching coordinates:** A corresponding original-resolution frame should be used for reliable overlays.
- **No webcam/live camera processing**, batch prediction pipeline, automatic temporal tracking, or annotated video export.
- **Checkpoint distribution is separate:** GitHub source alone is insufficient for new Stage 1 inference without `best.pt`.

## 8. Potential next steps

1. Add automatic processing of multiple video frames and export their combined keypoint sequence.
2. Improve saved-sequence browsing and annotated-video playback.
3. Connect Stage 2 temporal refinement and then Stage 3 recognition when those components are ready.
4. Evaluate selected landmark subsets in controlled experiments; visual filtering alone does not demonstrate an accuracy improvement.
5. Provide portable checkpoint delivery and reproducible dependency versions for teammates.

## 9. Contributor notes

- Work from the repository root so the `models.keypoints` imports resolve.
- Preserve the mapping from the **73 model output channels** to the **133 COCO-WholeBody IDs**.
- Preserve the distinction between an **illustrative reference skeleton**, **live Stage 1 predictions**, and **saved predictions**.
- Do not commit the local `.venv` directory or model checkpoints unintentionally.
- When distributing this app, explain that the checkpoint must be supplied separately and only from a trusted source.
- Verify both image and video-frame inference when modifying `infer_stage1()` or the plotting functions.
- Coordinate contributions to the shared `main` branch with group members.

## 10. Summary

The ASL Keypoint Explorer now integrates the team's **Stage 1 ResNet-50-based detector** with a Streamlit interface. It supports **133-landmark reference selection**, **real 73-point inference on uploaded ASL images and selected video frames**, **original-resolution video-frame export**, **prediction-array export/import**, **confidence filtering**, and **saved prediction visualization**. Stages 2 and 3 remain outside the app's current functionality.
