# ASL Keypoint Explorer — DATA 255 Group 2

An interactive **Streamlit interface** for exploring and customizing COCO-WholeBody landmark selections for our American Sign Language (ASL) keypoint-recognition research project.

**Current scope:** The application provides an interactive **133-point reference skeleton**, customizable landmark selections, image/video inspection, and visualization of previously generated keypoint predictions.

The application **does not currently run keypoint detection or ASL recognition on uploaded images or videos**. The default reference skeleton is illustrative, not a model prediction.

## 1. Purpose and Project Overview

Our ASL recognition pipeline consists of three stages:

1. **Stage 1 — Keypoint Detection:** A ResNet-50-based model predicts 73 ASL-relevant COCO-WholeBody keypoints.
2. **Stage 2 — Temporal Refinement:** A Transformer-based model refines unreliable or missing keypoints across video frames.
3. **Stage 3 — ASL Recognition:** A Bidirectional GRU (BiGRU) classifies word-level ASL signs from keypoint sequences.

The **ASL Keypoint Explorer** serves as a standalone visualization and customization interface for this pipeline, primarily focusing on Stage 1.

Its purpose is to allow users to:

- Explore the full COCO-WholeBody keypoint specification.
- Select specific landmarks or anatomical regions.
- Visualize selected landmarks on an interactive reference skeleton.
- Inspect ASL images and videos.
- Visualize previously generated model predictions.
- Export selected landmark indices for future experiments.

**Important:** The interface defines which landmarks users want to inspect. Actual keypoint detection and tracking require predictions from the trained detector.

Currently, selecting or deselecting landmarks affects **visualization and JSON export only**. It does not modify the trained detector, Transformer refiner, BiGRU classifier, or their prediction accuracy.

## 2. Installation and Setup

### Requirements

- Python 3.10 or newer
- The cloned `ASL_Project` repository
- A web browser
- The Python dependencies listed below

The application uses the existing repository module:

`models/keypoints/keypoint_spec.py`

Therefore, it should be launched from the repository root.

### Windows / PowerShell

Open PowerShell or the VS Code terminal and navigate to the project directory:

```powershell
cd "$HOME\Documents\ASL_Project"
```

Install the required packages:

```powershell
py -m pip install streamlit matplotlib numpy pillow opencv-python streamlit-image-coordinates
```

Launch the application:

```powershell
py -m streamlit run keypoint_explorer.py
```

If the script has a different filename, either rename it to `keypoint_explorer.py` or specify the actual filename in the command.

Streamlit should open automatically in your browser, usually at:

http://localhost:8501

If port 8501 is already occupied, Streamlit may select another available port.

To stop the application, press `Ctrl+C` in the terminal.

**Note:** Launch the application using `streamlit run`, not VS Code's ordinary **Run Python File** command. Running the script directly can produce Streamlit session-state warnings.

### Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web interface, tabs, session state, and file uploads |
| `streamlit-image-coordinates` | Interactive clicking on the skeleton |
| `matplotlib` | Landmark and skeleton visualization |
| `numpy` | Keypoint arrays and `.npy` prediction loading |
| `pillow` | Image processing and rendering |
| `opencv-python` | Video decoding and frame extraction |

**A GPU or trained model checkpoint is not required to run the standalone application.**

## 3. Application Features

The interface is organized into three tabs.

### Tab 1 — Keypoint Selector

This is the application's primary feature.

Users can customize which COCO-WholeBody landmarks are selected and visualize them on an interactive reference skeleton.

#### Selection Controls

Three primary options are provided:

- **All 133:** Select every COCO-WholeBody landmark.
- **Current 73:** Restore the 73 landmarks supported by the group's current model.
- **Clear:** Remove all selected landmarks.

#### Anatomical Presets

Users can add or remove predefined anatomical regions:

- Upper body
- Torso / hips
- Legs
- Feet
- Face
- Mouth
- Left hand
- Right hand

Presets may overlap. For example, the mouth landmarks are included within the full face region.

#### Individual Keypoint Selection

Users can browse and customize individual landmarks using three filters:

- **All 133:** Every COCO-WholeBody landmark.
- **Supported 73:** Landmarks supported by the current detector.
- **Future 60:** Landmarks not predicted by the current detector.

Individual landmarks can be added or removed without changing the other selected landmarks.

#### Interactive Reference Skeleton

The reference skeleton displays the selected landmarks using color-coded markers:

- **Green:** Body landmarks
- **Blue:** Left-hand landmarks
- **Orange:** Right-hand landmarks
- **Purple:** Facial and mouth landmarks

Users can:

- Click a landmark to select or deselect it.
- Show or hide landmark ID labels.
- Show or hide unselected landmarks.
- Enable a larger skeleton preview.
- Review the complete list of selected COCO IDs.

For closely spaced landmarks, such as fingers and facial features, the individual keypoint browser provides more precise selection.

**The skeleton uses illustrative coordinates. It is not generated from an uploaded image, video, or trained model.**

#### Selection Summary

The application displays:

- Total number of selected landmarks.
- Number supported by the current model.
- Number requiring expanded model support.

Selected unsupported landmarks are explicitly identified.

#### JSON Export

The current selection can be exported using the **Export selection (.json)** button.

This allows landmark configurations to be saved for future experiments or integration.

### Tab 2 — Image & Video

This tab allows users to upload and inspect ASL-related media.

#### Image Viewer

Supported formats:

- PNG
- JPG
- JPEG

Users can upload an image and display it directly in the application.

#### Video Viewer

Supported formats:

- MP4
- MOV
- AVI

Users can:

- Upload a video.
- Play the uploaded video.
- Select a frame using a slider.
- Inspect the decoded frame.

Video decoding depends on the codecs supported by the local OpenCV installation and browser.

**Current limitation:** Image and video uploads are for inspection only. The application does not automatically detect, track, or overlay keypoints on uploaded media.

### Tab 3 — Model Predictions

This tab allows users to visualize previously generated keypoint predictions.

#### Supported Prediction Formats

The application accepts NumPy `.npy` files containing arrays with either shape:

```python
(T, 73, 3)
```

or:

```python
(T, 133, 3)
```

Where:

- `T` = number of video frames
- `73` or `133` = number of landmarks
- `3` = `(x, y, confidence)`

For 73-landmark arrays, the application maps the model's output channels to the corresponding original COCO-WholeBody indices using `WHOLEBODY_INDICES`.

#### Prediction Visualization

Users can:

- Upload compatible prediction arrays.
- Select a prediction frame.
- Visualize predicted landmark positions.
- Display only the currently selected landmarks.
- Show or hide landmark IDs.
- Show or hide unselected landmarks.
- Adjust the minimum confidence threshold.
- Optionally upload a corresponding frame image as a background.

#### Confidence Filtering

The **Minimum confidence** slider determines which predicted landmarks are visible.

For example, with a minimum confidence of `0.50`, only selected landmarks whose confidence values are at least `0.50` are displayed.

This threshold affects **visualization only**.

It does not modify the Stage 2 Transformer's refinement threshold, change model predictions, or retrain the recognition system.

#### Coordinate Requirements

When overlaying predictions on a supplied image, the predicted `(x, y)` coordinates must match the image's coordinate system.

Coordinates generated from resized inputs, cropped frames, or normalized model outputs may require conversion before they align correctly.

The application does not currently perform this coordinate conversion automatically.

**Important:** This tab loads previously generated predictions. It does not execute `predict_wlasl.py` or run the trained CNN. Prediction files must already exist and use a compatible format.

## 4. COCO-WholeBody Keypoint Specification

COCO-WholeBody defines a total of **133 landmarks**.

| Region | Number of landmarks | COCO IDs |
|---|---:|---|
| Body | 17 | 0–16 |
| Feet | 6 | 17–22 |
| Face | 68 | 23–90 |
| Left hand | 21 | 91–111 |
| Right hand | 21 | 112–132 |
| **Total** | **133** | **0–132** |

### Current Model Support

Our group's current Stage 1 detector predicts **73 ASL-relevant landmarks**.

| Region | Number of landmarks | COCO IDs |
|---|---:|---|
| Upper body | 11 | 0–10 |
| Mouth | 20 | 71–90 |
| Left hand | 21 | 91–111 |
| Right hand | 21 | 112–132 |
| **Total** | **73** | |

The remaining **60 landmarks are not currently predicted** by Stage 1.

They can still be selected and displayed on the illustrative reference skeleton for exploration and future model development.

Selecting all 133 landmarks does not expand the trained model's output capabilities.

## 5. Exported Selection Format

Selecting **Export selection (.json)** downloads a file containing the selected original COCO-WholeBody indices.

Example:

```json
{
  "coco_wholebody_indices": [5, 6, 91, 112]
}
```

These indices represent the landmarks selected by the user.

The exported file does not contain:

- Predicted landmark coordinates.
- Video annotations.
- Model weights.
- Classification results.
- Changes to the trained model configuration.

The application does not currently support importing previously exported JSON configurations.

The selection is stored in Streamlit session state during the active browser session.

## 6. Quick Functional Testing Checklist

Before demonstrating or merging the application, verify the following functionality.

### Keypoint Selection

1. Click **Current 73** and verify that 73 landmarks are selected.
2. Click **All 133** and verify that 133 landmarks are selected, including 60 unsupported by the current model.
3. Click **Clear** and verify that the selection count becomes zero.
4. Add **Left hand** and verify that 21 landmarks are selected.
5. Add **Right hand** and verify that the total becomes 42.
6. Click a visible landmark to toggle its selection.
7. Click the same landmark again to reverse the selection.
8. Verify that individual keypoint filters preserve existing selections.
9. Export JSON and verify that the selected IDs match the file contents.

### Media Inspection

1. Upload a PNG or JPEG image.
2. Confirm that the image displays.
3. Upload a short MP4 or compatible video.
4. Verify video playback.
5. Move the frame slider and confirm that the displayed frame changes.

### Model Predictions

When compatible model-generated data is available:

1. Upload a valid `.npy` prediction file.
2. Verify that the application recognizes the array shape.
3. Select different prediction frames.
4. Change the minimum confidence threshold.
5. Confirm that low-confidence landmarks are hidden.
6. Optionally upload a corresponding background image.
7. Verify that predicted landmarks align with the image coordinates.

These checks are intended for manual validation. Python syntax validation alone does not establish that every browser interaction or prediction overlay works correctly.

## 7. Current Limitations

The following capabilities are **not yet implemented**:

- Automatically running Stage 1 keypoint detection on uploaded images or videos.
- Tracking newly detected landmarks throughout uploaded videos.
- Automatically overlaying newly inferred keypoints on uploaded media.
- Running Stage 2 temporal refinement from the Streamlit interface.
- Running Stage 3 ASL classification from the Streamlit interface.
- Predicting the 60 additional COCO-WholeBody landmarks.
- Applying arbitrary keypoint selections directly to the trained recognition model.
- Evaluating classification accuracy for different landmark subsets.
- Live webcam capture or real-time inference.

The application currently functions as a **standalone customization, visualization, and media-inspection interface**.

## 8. Planned Integration

The intended end goal is to allow users to upload an ASL image or video, choose their desired landmarks, and visualize those detected landmarks on the uploaded media.

### Milestone 1 — Stage 1 Integration

Connect the Streamlit application to the group's trained ResNet-50 keypoint detector.

This requires:

- Loading a compatible trained model checkpoint.
- Applying the same preprocessing used during training and inference.
- Running inference on uploaded images or video frames.
- Obtaining the detector's 73 supported landmark predictions.

### Milestone 2 — Customizable Keypoint Overlays

Use the current keypoint selection interface to filter detected landmarks.

Users should be able to:

- Select the desired landmarks or anatomical regions.
- Display only supported selected landmarks over uploaded frames.
- Identify selected landmarks that the model cannot predict.
- Adjust visualization settings.

The selection should control which landmarks are displayed, without modifying the model's learned parameters.

### Milestone 3 — Video Keypoint Tracking

Extend Stage 1 integration to video sequences.

Potential features include:

- Processing individual video frames.
- Visualizing selected landmarks across frames.
- Playing back annotated video.
- Exporting annotated results.

### Milestone 4 — Optional Research Experiments

Use exported landmark selections for reproducible experiments comparing different keypoint subsets.

Possible experiments include:

- Hands-only recognition.
- Hands and mouth recognition.
- Hands and upper-body recognition.
- Full 73-landmark recognition.

These experiments would require appropriate input masking or retraining strategies because the downstream models expect fixed-dimensional inputs.

Visualization filtering alone does not establish whether a landmark subset improves recognition accuracy.

## 9. Notes for Contributors

- The reference skeleton uses illustrative coordinates and should not be treated as actual pose-estimation output.
- The application imports the current model's keypoint mapping from `models/keypoints/keypoint_spec.py`.
- Original COCO-WholeBody landmark IDs and the model's 73 local output channels are different index systems; preserve their mapping.
- Landmark selections are maintained using Streamlit session state.
- Exported JSON provides a portable record of selected indices.
- The standalone application does not require model checkpoints or GPU acceleration.
- Changes should be tested on a feature branch before merging into the group's main branch.
- Model integration should preserve compatibility with the existing preprocessing, prediction formats, and downstream models.

## 10. Summary

The ASL Keypoint Explorer provides a user-friendly interface for **customizing and visualizing COCO-WholeBody keypoints** as part of the DATA 255 Group 2 ASL recognition project.

Its current functionality includes:

- Interactive selection of 133 COCO-WholeBody landmarks.
- Support for the group's existing 73-landmark configuration.
- Anatomical presets and individual landmark customization.
- Clickable reference skeleton visualization.
- Image and video inspection.
- Visualization of compatible precomputed model predictions.
- Confidence filtering and JSON selection export.

The next major development milestone is **integrating the trained Stage 1 detector with uploaded media**, enabling selected landmarks to be visualized on actual ASL images and video frames.