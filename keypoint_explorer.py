"""Interactive COCO-WholeBody keypoint selector for the DATA 255 ASL project.

Run from the repository root:
    python -m pip install streamlit matplotlib numpy pillow opencv-python streamlit-image-coordinates
    python -m streamlit run keypoint_explorer.py

This app can run the trained 73-point Stage 1 detector if its checkpoint is available.
"""
from __future__ import annotations

import io
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import cv2
import tempfile
import os
import sys

from models.keypoints.keypoint_spec import WHOLEBODY_INDICES

# Stage 1: reuse the team's extractor implementation and checkpoint.
ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "runs" / "keypoints_resnet50_256x192" / "best.pt"


@st.cache_resource(show_spinner="Loading Stage 1 ResNet-50 model...")
def load_stage1():
    import torch

    model_dir = str(ROOT / "models" / "keypoints")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from predict_wlasl import load_model, predict_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The team's loader uses torch.load(weights_only=False), so load only
    # checkpoints obtained from a trusted teammate.
    model, args = load_model(str(CHECKPOINT), device)
    return model, args, device, predict_clip


def infer_stage1(rgb: np.ndarray) -> np.ndarray:
    """Run Stage 1 on one RGB image; return 73 rows of (x, y, score)."""
    model, args, device, predict_clip = load_stage1()
    output = predict_clip(model, rgb[None, ...], args["input_h"], args["input_w"], device)
    if output.shape != (1, len(WHOLEBODY_INDICES), 3):
        raise ValueError(f"Unexpected Stage 1 output shape: {output.shape}")
    return output[0]


def wholebody_points(kpts73: np.ndarray) -> np.ndarray:
    all_points = np.full((133, 3), np.nan, dtype=float)
    all_points[np.asarray(WHOLEBODY_INDICES)] = kpts73
    return all_points


def npy_bytes(kpts73: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, kpts73[None, ...])
    return buf.getvalue()


SUPPORTED = set(map(int, WHOLEBODY_INDICES))
ALL = set(range(133))
GROUPS = {
    "Face (68)": set(range(23, 91)),
    "Mouth (20)": set(range(71, 91)),
    "Upper body (11)": set(range(0, 11)),
    "Torso / hips (2)": {11, 12},
    "Legs (4)": {13, 14, 15, 16},
    "Feet (6)": set(range(17, 23)),
    "Left hand (21)": set(range(91, 112)),
    "Right hand (21)": set(range(112, 133)),
}
COCO_NAMES = [
    "nose", "left eye", "right eye", "left ear", "right ear",
    "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left hip", "right hip",
    "left knee", "right knee", "left ankle", "right ankle",
    "left big toe", "left small toe", "left heel",
    "right big toe", "right small toe", "right heel",
]

def label(i: int) -> str:
    if i < 23:
        return f"{i:03d} · {COCO_NAMES[i]}"
    if i < 91:
        return f"{i:03d} · face landmark {i-23}"
    if i < 112:
        return f"{i:03d} · left hand landmark {i-91}"
    return f"{i:03d} · right hand landmark {i-112}"


def template() -> np.ndarray:
    """Illustrative front-facing pose; not predictions or annotated data."""
    p = np.full((133, 2), np.nan, dtype=float)
    p[:23] = [
        (0, 2.75), (-.11, 2.79), (.11, 2.79), (-.23, 2.74), (.23, 2.74),
        (-.42, 2.33), (.42, 2.33), (-.59, 1.93), (.59, 1.93),
        (-.72, 1.59), (.72, 1.59), (-.28, 1.41), (.28, 1.41),
        (-.30, .83), (.30, .83), (-.30, .24), (.30, .24),
        (-.37, .12), (-.25, .12), (-.30, .15),
        (.37, .12), (.25, .12), (.30, .15),
    ]
    # Approximate layout of the standard 68 facial landmark indices.
    cx, cy = 0.0, 2.70
    theta = np.linspace(np.pi*1.04, np.pi*-.04, 17)
    p[23:40, 0] = cx + .27 * np.cos(theta)
    p[23:40, 1] = cy - .09 + .27 * np.sin(theta)
    brow_x = np.linspace(-.22, -.055, 5)
    p[40:45] = list(zip(brow_x, [2.87,2.90,2.91,2.90,2.88]))
    brow_x = np.linspace(.055, .22, 5)
    p[45:50] = list(zip(brow_x, [2.88,2.90,2.91,2.90,2.87]))
    p[50:54] = [(0,2.82),(0,2.76),(0,2.70),(0,2.65)]
    p[54:59] = [(-.07,2.63),(-.04,2.61),(0,2.60),(.04,2.61),(.07,2.63)]
    for start, ex in [(59,-.12),(65,.12)]:
        a = np.linspace(0,2*np.pi,6, endpoint=False)
        p[start:start+6] = np.c_[ex+.055*np.cos(a), 2.76+.027*np.sin(a)]
    # Mouth indices 71-90, in two nested ovals.
    a = np.linspace(0,2*np.pi,12,endpoint=False)
    p[71:83] = np.c_[.125*np.cos(a),2.51+.055*np.sin(a)]
    a = np.linspace(0,2*np.pi,8,endpoint=False)
    p[83:91] = np.c_[.07*np.cos(a),2.51+.025*np.sin(a)]

    def hand(start, wrist_x, wrist_y, direction):
        p[start] = (wrist_x,wrist_y)
        # COCO hand order: wrist then 4 joints per finger (thumb/index/middle/ring/pinky).
        for finger in range(5):
            span = (finger-2)*.07
            for joint in range(4):
                i = start+1+4*finger+joint
                reach = (joint+1)*.077
                p[i] = (wrist_x + direction*(.12+reach) + span,
                        wrist_y + .04 + (2-finger)*.034 + (joint+1)*.022)
    hand(91,-.72,1.59,-1)
    hand(112,.72,1.59,1)
    return p

EDGES = [
    (5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),(15,17),(15,18),(15,19),
    (16,20),(16,21),(16,22),(0,1),(0,2),(1,3),(2,4),
    (9,91),(10,112),
]
for base in (91,112):
    for finger in range(5):
        seq = [base]+[base+1+4*finger+k for k in range(4)]
        EDGES.extend(zip(seq,seq[1:]))


def color_for(i: int) -> str:
    if 91 <= i < 112: return '#3b82f6'
    if 112 <= i <= 132: return '#e97142'
    if 71 <= i < 91: return '#be4bdb'
    if 23 <= i < 71: return '#a78bfa'
    return '#22a06b'


def draw(points: np.ndarray, selected: set[int], show_ids: bool,
         ghost: bool, confidence: float, background=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('#0e1625')
    ax.set_facecolor('#0e1625')
    
    if background is not None:
        ax.imshow(background)
        ax.set_xlim(0, background.shape[1])
        ax.set_ylim(background.shape[0], 0)
        ax.set_aspect("equal")

    elif points.shape[1] == 3:
        # Prediction data: automatically fit the available coordinates
        valid = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])

        if np.any(valid):
            xy = points[valid, :2]
            x_min, y_min = xy.min(axis=0)
            x_max, y_max = xy.max(axis=0)

            pad = max(x_max - x_min, y_max - y_min, 1) * 0.1

            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_max + pad, y_min - pad)
        else:
            ax.set_xlim(0, 224)
            ax.set_ylim(224, 0)

        ax.set_aspect("equal")

    else:
        # Original illustrative reference skeleton
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.1, 3.2)
        ax.set_aspect("equal")

    if ghost:
        for a,b in EDGES:
            if np.all(np.isfinite(points[[a,b], :2])):
                ax.plot(points[[a,b],0],points[[a,b],1], color='#68768b', alpha=.23, linewidth=1)
        valid = np.isfinite(points[:,0]) & np.isfinite(points[:,1])
        ax.scatter(points[valid,0],points[valid,1],s=8,c='#68768b',alpha=.3)
    active = {i for i in selected if np.all(np.isfinite(points[i,:2]))
              and (points.shape[1]<3 or points[i,2]>=confidence)}
    for a,b in EDGES:
        if a in active and b in active:
            ax.plot(points[[a,b],0],points[[a,b],1],color=color_for(a),alpha=.7,linewidth=1.7,zorder=3)
    for i in sorted(active):
        x,y=points[i,:2]
        ax.scatter([x],[y],c=[color_for(i)],s=28,edgecolor='white',linewidth=.4,zorder=4)
        if show_ids:
            ax.annotate(
                str(i),
                (x, y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5,
                color="white",
                fontweight="bold",
                bbox=dict(
                    facecolor="#0e1625",
                    edgecolor="none",
                    alpha=0.8,
                    pad=0.5
                ),
                zorder=5
            )
    ax.axis('off')
    fig.tight_layout(pad=.1)
    return fig, len(active)


st.set_page_config(page_title="ASL Keypoint Explorer", layout="wide")
st.title("ASL Keypoint Explorer")
st.caption("DATA 255 · COCO-WholeBody · 73 currently modeled / 133 defined")
st.info(
    "The reference skeleton is illustrative, not model output. The trained CNN predicts "
    "73 of 133 keypoints; the other 60 are selectable for planning future support."
)

if "selection" not in st.session_state:
    st.session_state.selection = sorted(SUPPORTED)

# Apply clicks BEFORE the multiselect widget is instantiated.
if "pending_diagram_toggle" in st.session_state:
    point_id = st.session_state.pop("pending_diagram_toggle")
    selected_now = set(st.session_state.selection)
    if point_id in selected_now:
        selected_now.remove(point_id)
    else:
        selected_now.add(point_id)
    st.session_state.selection = sorted(selected_now)


def set_selection(value):
    st.session_state.selection = sorted(value)


def add_selection(value):
    set_selection(set(st.session_state.selection) | set(value))


def remove_selection(value):
    set_selection(set(st.session_state.selection) - set(value))


def render_skeleton(points, *, widget_prefix, background=None, predicted=False):
    """Render an interactive skeleton; clicks toggle COCO landmark IDs."""
    if predicted:
        a, b, c = st.columns(3)
        show_ids = a.checkbox("Show point IDs", key=f"{widget_prefix}_ids")
        ghost = b.checkbox("Show unselected", value=True, key=f"{widget_prefix}_ghost")
        confidence = c.slider(
            "Min confidence", 0.0, 1.0, 0.2, 0.05,
            key=f"{widget_prefix}_confidence",
            help="Show only predictions whose detector confidence meets this threshold."
        )
    else:
        a, b = st.columns(2)
        show_ids = a.checkbox("Show point IDs", key=f"{widget_prefix}_ids")
        ghost = b.checkbox("Show unselected", value=True, key=f"{widget_prefix}_ghost")
        confidence = 0.0  # Illustrative points have no confidence scores.
    st.caption("Click a dot to toggle it; for crowded facial and finger landmarks, "
               "use the individual landmark browser on the Keypoint Selector tab.")
    large_view = st.toggle("Large skeleton view", key=f"{widget_prefix}_large")
    display_width = 650 if large_view else 400

    fig, visible = draw(points, set(st.session_state.selection),
                        show_ids, ghost, confidence, background)
    # Use the exact raster dimensions that the click-coordinate component sees.
    fig.set_dpi(220)
    fig.canvas.draw()
    native_width, native_height = fig.canvas.get_width_height()
    landmark_pixels = {}
    for pid in range(133):
        if np.all(np.isfinite(points[pid, :2])):
            px, py = fig.axes[0].transData.transform(points[pid, :2])
            landmark_pixels[pid] = (px, native_height - py)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    picture = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    display_height = round(picture.height * display_width / picture.width)
    picture = picture.resize((display_width, display_height), Image.Resampling.LANCZOS)

    click = streamlit_image_coordinates(picture, key=f"{widget_prefix}_clickable_skeleton")
    if click is not None:
        event_id = (click.get("unix_time"), click.get("x"), click.get("y"))
        last_click_key = f"{widget_prefix}_last_click"
        if event_id != st.session_state.get(last_click_key):
            st.session_state[last_click_key] = event_id
            x, y = float(click["x"]), float(click["y"])
            scale = display_width / native_width
            distances = [
                (np.hypot(x - px * scale, y - py * scale), pid)
                for pid, (px, py) in landmark_pixels.items()
            ]
            if distances:
                distance, nearest_id = min(distances)
                if distance <= (18 if large_view else 12):
                    st.session_state.pending_diagram_toggle = nearest_id
                    st.rerun()
                else:
                    st.caption("Click closer to a landmark dot to select it.")
    st.caption(f"{visible} selected landmarks visible · " +
               ("Prediction confidence filter active" if predicted else "Reference geometry only"))
    with st.expander("Selected COCO IDs", expanded=False):
        st.write(", ".join(map(str, sorted(st.session_state.selection))) or "None")


tab1, tab2, tab3 = st.tabs([
    "Keypoint Selector", "Image & Video", "Model Predictions"
])

with tab1:
    left, right = st.columns([1, 1], gap="medium")
    with left:
        st.subheader("Selection controls")
        c1, c2, c3 = st.columns(3)
        c1.button("All 133", on_click=set_selection, args=(ALL,), use_container_width=True)
        c2.button("Current 73", on_click=set_selection, args=(SUPPORTED,), use_container_width=True)
        c3.button("Clear", on_click=set_selection, args=(set(),), use_container_width=True)

        with st.expander("Anatomical presets", expanded=False):
            for name, ids in GROUPS.items():
                a, b, c = st.columns([2.2, 1, 1])
                a.caption(f"{name} ({len(ids & set(st.session_state.selection))}/{len(ids)})")
                b.button("Add", key="add_" + name, on_click=add_selection,
                         args=(ids,), use_container_width=True)
                c.button("Remove", key="remove_" + name, on_click=remove_selection,
                         args=(ids,), use_container_width=True)

        with st.expander("Browse individual keypoints", expanded=False):
            point_filter = st.radio(
                "Browse keypoints", ["All 133", "Supported 73", "Future 60"],
                horizontal=True
            )
            if point_filter == "Supported 73":
                filtered_ids = sorted(SUPPORTED)
            elif point_filter == "Future 60":
                filtered_ids = sorted(ALL - SUPPORTED)
            else:
                filtered_ids = sorted(ALL)
            point_to_edit = st.selectbox("Find a keypoint", options=filtered_ids,
                                         format_func=label)
            add_col, remove_col = st.columns(2)
            add_col.button("Add selected point", key="add_single",
                           on_click=add_selection, args=({point_to_edit},),
                           use_container_width=True)
            remove_col.button("Remove selected point", key="remove_single",
                              on_click=remove_selection, args=({point_to_edit},),
                              use_container_width=True)

        with st.expander("Edit complete selected-ID list", expanded=False):
            st.multiselect("All selected keypoints", options=list(range(133)),
                           format_func=label, key="selection")
        # The selection multiselect remains instantiated even when the expander is collapsed.
        chosen = set(st.session_state.selection)
        modeled = chosen & SUPPORTED
        unavailable = chosen - SUPPORTED
        st.metric("Selected", f"{len(chosen)} / 133")
        st.caption(f"{len(modeled)} selected model-supported · {len(unavailable)} require model expansion")
        if unavailable:
            with st.expander(f"{len(unavailable)} selected unsupported points", expanded=False):
                st.warning("Reference only, not predicted by the current CNN: " +
                           ", ".join(map(str, sorted(unavailable))))
        st.download_button(
            "Export selection (.json)",
            data=__import__("json").dumps({"coco_wholebody_indices": sorted(chosen)}, indent=2),
            file_name="keypoint_selection.json", mime="application/json"
        )

    with right:
        st.subheader("Interactive reference skeleton")
        st.caption("Blue: left hand · Orange: right hand · Green: body · Purple: face/mouth")
        render_skeleton(template(), widget_prefix="reference")

with tab2:
    st.subheader("ASL Image & Video Explorer")
    st.caption("Upload media and optionally run the trained Stage 1 model to visualize its 73 keypoints.")
    if CHECKPOINT.is_file():
        st.success("Stage 1 checkpoint found locally.")
    else:
        st.info(f"Stage 1 checkpoint not found at: {CHECKPOINT}. Media viewing still works.")
    source = st.radio("Media type", ["Image", "Video"], horizontal=True)
    if source == "Image":
        uploaded_image = st.file_uploader(
            "Upload an ASL image", type=["png", "jpg", "jpeg"], key="asl_image"
        )
        if uploaded_image is not None:
            rgb_image = np.asarray(Image.open(uploaded_image).convert("RGB"))
            st.image(rgb_image, caption="Uploaded image", width=500)
            if st.button("Detect Stage 1 keypoints in image", disabled=not CHECKPOINT.is_file()):
                try:
                    with st.spinner("Running Stage 1 image inference..."):
                        st.session_state["stage1_image_result"] = {
                            "upload_key": (uploaded_image.name, uploaded_image.size),
                            "points": infer_stage1(rgb_image),
                        }
                except Exception as exc:
                    st.error(f"Stage 1 image inference failed: {exc}")
            result = st.session_state.get("stage1_image_result")
            if result is not None and result["upload_key"] == (uploaded_image.name, uploaded_image.size):
                st.subheader("Stage 1 image predictions (73 keypoints)")
                render_skeleton(wholebody_points(result["points"]),
                                widget_prefix="stage1_image", background=rgb_image, predicted=True)
                st.download_button("Download image predictions (.npy)",
                                   data=npy_bytes(result["points"]),
                                   file_name="image_stage1_predictions.npy",
                                   mime="application/octet-stream")
    else:
        video = st.file_uploader("Upload an ASL video", type=["mp4", "mov", "avi"],
                                 key="asl_video")
        if video is not None:
            st.video(video)
            suffix = os.path.splitext(video.name)[1]
            path = None
            cap = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                    temp.write(video.getvalue())
                    path = temp.name
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    st.error("The uploaded video could not be opened by OpenCV.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames > 0:
                        frame_idx = st.slider("Select frame", 0, total_frames - 1, 0,
                                              key="media_frame")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        success, frame = cap.read()
                        if success:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(rgb_frame, caption=f"Frame {frame_idx + 1}/{total_frames}",
                                     width=500)
                            if st.button("Detect Stage 1 keypoints in this frame",
                                         disabled=not CHECKPOINT.is_file()):
                                try:
                                    with st.spinner("Running Stage 1 frame inference..."):
                                        st.session_state["stage1_video_result"] = {
                                            "upload_key": (video.name, video.size, frame_idx),
                                            "points": infer_stage1(rgb_frame),
                                        }
                                except Exception as exc:
                                    st.error(f"Stage 1 video-frame inference failed: {exc}")
                            result = st.session_state.get("stage1_video_result")
                            if result is not None and result["upload_key"] == (video.name, video.size, frame_idx):
                                st.subheader(f"Stage 1 predictions — frame {frame_idx + 1}")
                                render_skeleton(wholebody_points(result["points"]),
                                                widget_prefix="stage1_video", background=rgb_frame,
                                                predicted=True)
                                st.download_button("Download frame predictions (.npy)",
                                                   data=npy_bytes(result["points"]),
                                                   file_name=f"frame_{frame_idx + 1}_stage1.npy",
                                                   mime="application/octet-stream")
                        else:
                            st.error("Could not decode the selected frame.")
                    else:
                        st.error("Could not determine the video frame count.")
            finally:
                if cap is not None:
                    cap.release()
                if path is not None and os.path.exists(path):
                    os.remove(path)

with tab3:
    st.subheader("Model Prediction Viewer")
    st.caption("Load previously generated keypoint coordinates. This tab does not run the CNN.")
    uploaded = st.file_uploader("Keypoint sequence (.npy)", type=["npy"],
                                key="model_npy")
    frame_img = st.file_uploader("Corresponding frame image (optional)",
                                 type=["png", "jpg", "jpeg"], key="model_image")
    if uploaded is None:
        st.info("Upload a .npy file shaped (T, 73, 3) to view Stage 1 predictions, "
                "or (T, 133, 3) for a future detector.")
    else:
        try:
            arr = np.load(io.BytesIO(uploaded.getvalue()), allow_pickle=False)
            if arr.ndim != 3 or arr.shape[1] not in (73, 133) or arr.shape[2] != 3:
                raise ValueError(f"Expected (T, 73, 3) or (T, 133, 3); got {arr.shape}")
            if arr.shape[0] == 0:
                raise ValueError("The sequence contains no frames.")
            
            if arr.shape[0] == 1:
                fi = 0
                st.caption("Single-frame prediction loaded.")
            else:
                fi = st.slider(
                    "Prediction frame",
                    min_value=0,
                    max_value=arr.shape[0] - 1,
                    value=0,
                    key="prediction_frame"
                )

            points = np.full((133, 3), np.nan, dtype=float)
            if arr.shape[1] == 73:
                points[np.asarray(WHOLEBODY_INDICES)] = arr[fi]
            else:
                points[:] = arr[fi]
            background = None
            if frame_img is not None:
                background = np.asarray(Image.open(io.BytesIO(frame_img.getvalue())).convert("RGB"))
            st.success(f"Loaded prediction frame {fi + 1} of {arr.shape[0]}")
            render_skeleton(points, widget_prefix="prediction", background=background,
                            predicted=True)
        except (ValueError, OSError) as exc:
            st.error(f"Could not load predictions: {exc}")

st.divider()
st.caption("Keypoint selection currently changes visualization and JSON export only; "
           "it does not alter trained model inputs or enable inference for the other 60 landmarks.")