"""Streamlit tab: ASL word recognition with Stages 1 -> 2 -> 3, plus temporal analysis.

Rendered from keypoint_explorer.py. Three inputs:
  * a WLASL test clip from the original run (stored keypoints, any degradation
    condition; no video needed, and the true word is known),
  * an uploaded video,
  * the last clip recorded in the Live Camera tab.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from contextlib import nullcontext

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from models.asl_pipeline import (
    CONDITIONS, MODEL_CONFIGS, NUM_FRAMES, PRIMARY_MODEL, Recognizer, correction_per_frame,
    draw_keypoints, group_confidence, hand_motion, load_labels, load_stored_keypoints, missing_files,
    read_video_frames, recognize_frames, stored_clips,
)

RESULT_KEY = "word_recognition_result"
GROUP_NAMES = {"upper_body": "Upper body", "mouth": "Mouth",
               "left_hand": "Left hand", "right_hand": "Right hand"}
SOURCES = ("WLASL test clip (stored keypoints)", "Upload a video", "Webcam recording")
CROP_LABELS = {"auto": "Auto signer crop (recommended)", "letterbox": "Whole frame (letterbox)"}


@st.cache_resource(show_spinner="Loading Stage 2 refiners and Stage 3 classifiers...")
def load_recognizer() -> Recognizer:
    return Recognizer()


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype=np.float32))
    return buf.getvalue()


def _run_on_frames(frames_rgb, load_stage1, lock, crop, source_label):
    stage1 = load_stage1()
    with (lock or nullcontext()):
        result = recognize_frames(frames_rgb, stage1, load_recognizer(), crop=crop)
    result.update(truth=None, source=source_label)
    st.session_state[RESULT_KEY] = result


def render_word_recognition(load_stage1, stage1_available: bool, inference_lock=None,
                            media_generation: int = 0) -> None:
    st.subheader("ASL word recognition: Stage 1 → Stage 2 → Stage 3")
    st.caption(f"{NUM_FRAMES} evenly spaced frames → Stage 1 keypoints (73 points) → "
               "Stage 2 temporal refiner (t−2…t+2 window) → Stage 3 Bi-GRU → one of 100 WLASL words.")

    missing = missing_files()
    if missing:
        st.error("Stage 2/3 files not found:\n\n" + "\n".join(f"- `{p}`" for p in missing))
        st.info("Copy the team's runs/pipeline folder (refiners, classifiers and "
                "keypoints/clean/manifest_keypoints.csv) into this repo's runs/ folder.")
        return

    _word_list()

    source = st.radio("Input", SOURCES, horizontal=True, key="wr_source")

    if source == SOURCES[0]:
        _stored_clip_input()
    elif source == SOURCES[1]:
        if not stage1_available:
            st.error("The Stage 1 checkpoint is needed to process video.")
        else:
            uploaded = st.file_uploader("ASL video", type=["mp4", "mov", "avi", "m4v", "webm"],
                                        key=f"wr_video_{media_generation}")
            crop = st.radio("Signer crop", list(CROP_LABELS), format_func=CROP_LABELS.get,
                            horizontal=True, key="wr_crop_upload",
                            help="The models were trained on square crops centred on the signer. "
                                 "Auto runs Stage 1 once on the full frames to find the signer.")
            if uploaded is not None and st.button("Recognize sign", type="primary", key="wr_go_upload"):
                suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    path = tmp.name
                try:
                    with st.spinner("Running Stages 1–3..."):
                        frames = read_video_frames(path, NUM_FRAMES)
                        _run_on_frames(frames, load_stage1, inference_lock, crop, uploaded.name)
                except Exception as exc:
                    st.error(f"Could not process the video: {exc}")
                finally:
                    os.unlink(path)
    else:
        clip = st.session_state.get("webcam_last_clip")
        if not stage1_available:
            st.error("The Stage 1 checkpoint is needed to process recordings.")
        elif clip is None or not clip.get("frames"):
            st.info("Record a clip in the Live Camera tab (Record clip → Stop recording / save clip), "
                    "then come back here.")
        else:
            st.caption(f"Last recording: {len(clip['frames'])} frames. "
                       f"{NUM_FRAMES} evenly spaced frames will be used.")
            crop = st.radio("Signer crop", list(CROP_LABELS), format_func=CROP_LABELS.get,
                            horizontal=True, key="wr_crop_webcam")
            if st.button("Recognize recording", type="primary", key="wr_go_webcam"):
                frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip["frames"]]
                try:
                    with st.spinner("Running Stages 1–3..."):
                        _run_on_frames(frames, load_stage1, inference_lock, crop, "webcam recording")
                except Exception as exc:
                    st.error(f"Could not process the recording: {exc}")

    result = st.session_state.get(RESULT_KEY)
    if result is not None:
        st.divider()
        _show_result(result)


def _word_list() -> None:
    """The 100 WLASL100 words the classifier knows, alphabetically, with a search box."""
    words = sorted(load_labels())
    with st.expander(f"The {len(words)} words it can recognize"):
        st.caption("Any sign outside this list is still mapped to the closest of these words. "
                   "Sign the standard dictionary form, one word per clip.")
        query = st.text_input("Search", key="wr_word_search", placeholder="e.g. computer")
        shown = [w for w in words if query.strip().lower() in w.lower()] if query else words
        if not shown:
            st.warning(f"'{query}' is not one of the {len(words)} words.")
            return
        n_cols = 5
        cols = st.columns(n_cols)
        per_col = -(-len(shown) // n_cols)
        for i, col in enumerate(cols):
            col.markdown("\n".join(f"- {w}" for w in shown[i * per_col:(i + 1) * per_col]))


def _stored_clip_input() -> None:
    clips = sorted(stored_clips("test"), key=lambda r: (r["gloss"], r["video_id"]))
    labels = [f"{r['gloss']} · clip {r['video_id']}" for r in clips]
    col_clip, col_cond = st.columns([2, 1])
    idx = col_clip.selectbox("Test clip (true word · WLASL video id)", range(len(clips)),
                             format_func=labels.__getitem__, key="wr_stored_clip")
    cond = col_cond.selectbox("Condition", CONDITIONS, key="wr_condition",
                              help="Seeded degradations from the original GPU run, so every model "
                                   "sees exactly the same damaged frames.")
    st.caption("Uses the Stage 1 keypoints saved by the original GPU run (161 test clips), "
               "so no video is needed. Good for demos: the true word is known.")
    if st.button("Recognize clip", type="primary", key="wr_go_stored"):
        row = clips[idx]
        raw = load_stored_keypoints(row["video_id"], cond)
        st.session_state[RESULT_KEY] = {
            "clip": None, "box": None, "stage1": raw, "models": load_recognizer().run(raw),
            "truth": row["gloss"], "source": f"WLASL clip {row['video_id']} ({cond})",
        }


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def _show_result(res: dict) -> None:
    models = res["models"]
    primary = models[PRIMARY_MODEL]
    word, prob = primary["top5"][0]
    truth = res.get("truth")

    st.markdown(f"**Input:** {res['source']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted word (Proposed)", word)
    c2.metric("Confidence", f"{prob:.0%}")
    if truth is not None:
        in_top5 = truth in [w for w, _ in primary["top5"]]
        c3.metric("True word", truth,
                  delta="correct" if word == truth else ("in top-5" if in_top5 else "missed"),
                  delta_color="normal" if word == truth else ("off" if in_top5 else "inverse"))
    st.caption("Clean-test accuracy of the Proposed model in the original run: 32.3% top-1, "
               "60.2% top-5 over 100 words, so check the top-5 list, not just the top word.")

    st.markdown("#### Top-5 words by model")
    cols = st.columns(len(models))
    for col, (name, out) in zip(cols, models.items()):
        df = pd.DataFrame(out["top5"], columns=["word", "probability"])
        df["probability"] = (df["probability"] * 100).round(1).astype(str) + "%"
        if truth is not None:
            df["word"] = [f"✅ {w}" if w == truth else w for w in df["word"]]
        df.index = range(1, len(df) + 1)
        col.markdown(f"**{name}**")
        col.dataframe(df, width="stretch")

    _temporal_analysis(res)
    _frame_viewer(res)

    raw, refined = res["stage1"], primary["keypoints"]
    d1, d2, d3 = st.columns(3)
    d1.download_button("Stage 1 keypoints (.npy)", _npy_bytes(raw),
                       "stage1_keypoints.npy", "application/octet-stream", key="wr_dl_raw")
    d2.download_button("Stage 2 refined keypoints (.npy)", _npy_bytes(refined),
                       "stage2_refined_keypoints.npy", "application/octet-stream", key="wr_dl_ref")
    d3.download_button(
        "Predictions (.json)",
        json.dumps({"source": res["source"], "true_word": truth,
                    "models": {n: o["top5"] for n, o in models.items()}}, indent=2),
        "predictions.json", "application/json", key="wr_dl_json")


def _temporal_analysis(res: dict) -> None:
    raw = res["stage1"]
    refined = res["models"][PRIMARY_MODEL]["keypoints"]
    frames = np.arange(1, raw.shape[0] + 1)

    st.markdown("#### Temporal analysis")
    st.caption("How keypoint quality changes across the clip, and what Stage 2 does about it. "
               "x-axis: sampled frame (1–16).")

    conf_raw, conf_ref = group_confidence(raw), group_confidence(refined)
    left, right = st.columns(2)
    left.markdown("**Stage 1 confidence by body part**")
    left.line_chart(pd.DataFrame({GROUP_NAMES[g]: v for g, v in conf_raw.items()}, index=frames),
                    y_label="mean confidence", x_label="frame")
    right.markdown("**After Stage 2 refinement**")
    right.line_chart(pd.DataFrame({GROUP_NAMES[g]: v for g, v in conf_ref.items()}, index=frames),
                     y_label="mean confidence", x_label="frame")

    left, right = st.columns(2)
    left.markdown("**Stage 2 correction per frame**")
    left.line_chart(pd.DataFrame(correction_per_frame(raw, refined), index=frames),
                    y_label="mean move (px in 224 frame)", x_label="frame")
    left.caption("Spikes mark frames where Stage 2 repaired weak points using neighbouring frames "
                 "(e.g. occluded or missing frames).")

    hand = right.radio("Hand trajectory", ["right_hand", "left_hand"], format_func=GROUP_NAMES.get,
                       horizontal=True, key="wr_hand")
    m_raw, m_ref = hand_motion(raw)[hand], hand_motion(refined)[hand]
    right.line_chart(pd.DataFrame({"x Stage 1": m_raw[:, 0], "x Stage 2": m_ref[:, 0],
                                   "y Stage 1": m_raw[:, 1], "y Stage 2": m_ref[:, 1]},
                                  index=frames),
                     y_label="hand centre (px)", x_label="frame")
    right.caption("Hand centre over time (signer's own left/right). The sign's motion is what "
                  "Stage 3 reads; Stage 2 should smooth jumps without flattening real motion.")


def _frame_viewer(res: dict) -> None:
    raw = res["stage1"]
    refined = res["models"][PRIMARY_MODEL]["keypoints"]
    clip = res.get("clip")
    st.markdown("#### Frame by frame")
    t = st.slider("Frame", 1, raw.shape[0], 1, key="wr_frame") - 1
    min_score = st.slider("Minimum confidence to draw", 0.0, 1.0, 0.2, 0.05, key="wr_min_score")
    base = clip[t] if clip is not None else np.zeros((224, 224, 3), np.uint8)
    a, b = st.columns(2)
    a.image(draw_keypoints(base, raw[t], min_score), caption=f"Stage 1 · frame {t + 1}",
            width="stretch")
    b.image(draw_keypoints(base, refined[t], min_score), caption=f"Stage 2 refined · frame {t + 1}",
            width="stretch")
    if clip is None:
        st.caption("Stored clips have keypoints only (WLASL frames are kept private), "
                   "so points are drawn on a blank 224×224 frame.")
    elif res.get("box") is None:
        st.caption("No signer box found; the whole frame was letterboxed.")
