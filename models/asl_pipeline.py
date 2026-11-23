"""Stages 1-3 end to end, for the Streamlit app.

    video frames -> 16 sampled, signer-centred 224x224 frames
                 -> Stage 1 keypoints (73 points, pixel coords in the 224 frame)
                 -> Stage 2 temporal refiner (t-2..t+2 window)
                 -> Stage 3 Bi-GRU classifier -> one of 100 WLASL glosses

Everything here mirrors how the models were trained and evaluated
(models/run_pipeline.sh, models/evaluate_robustness.py):
  * Stage 3 reads keypoints in the 224x224 frame, normalised to [-0.5, 0.5].
  * "Proposed" = refiner_degraded -> classifier_refined_degraded.
  * "+Temporal" = refiner_synthetic -> classifier_refined_synthetic.
  * "Frame-only" = classifier_frame_only on raw Stage 1 keypoints.

Stages 2 and 3 are tiny (a few MB) and run on the CPU; that is faster than
moving 16 frames to a GPU and avoids Apple-MPS operator gaps.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("models/keypoints", "models/refinement", "models/recognition", "data/wlasl"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from classifier import SignClassifier  # noqa: E402
from keypoint_sequences import FEATURES_PER_FRAME, normalize_keypoints  # noqa: E402
from keypoint_spec import GROUP_SLICES  # noqa: E402
from refiner import TemporalRefiner  # noqa: E402
from sample_frames import even_indices, letterbox, square_crop  # noqa: E402
from windows import make_windows, to_model_space, to_pixel_space  # noqa: E402

PIPELINE_DIR = ROOT / "runs" / "pipeline"
LABEL_MANIFEST = PIPELINE_DIR / "keypoints" / "clean" / "manifest_keypoints.csv"
STORED_KEYPOINTS = PIPELINE_DIR / "keypoints"

NUM_FRAMES = 16
FRAME_SIZE = 224

# Display name -> (classifier folder, refiner folder or None)
MODEL_CONFIGS = {
    "Proposed (+Temporal & robust training)": ("classifier_refined_degraded", "refiner_degraded"),
    "+Temporal (synthetic-noise refiner)": ("classifier_refined_synthetic", "refiner_synthetic"),
    "Frame-only (no Stage 2)": ("classifier_frame_only", None),
}
PRIMARY_MODEL = "Proposed (+Temporal & robust training)"

CONDITIONS = ["clean"] + [f"{d}_{s}" for d in ("occlusion", "blur", "missing")
                          for s in ("mild", "moderate", "severe")]

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _checkpoint(path: Path) -> dict:
    # Checkpoints were saved on CUDA; map to CPU so they load on any machine.
    # weights_only=False: only load checkpoints from trusted teammates.
    return torch.load(str(path), map_location="cpu", weights_only=False)


def load_refiner(path: Path) -> TemporalRefiner:
    ckpt = _checkpoint(path)
    a = ckpt["args"]
    model = TemporalRefiner(arch=a["arch"], hidden=a["hidden"], num_layers=a["num_layers"])
    model.load_state_dict(ckpt["model"])
    return model.eval()


def load_classifier(path: Path) -> SignClassifier:
    ckpt = _checkpoint(path)
    a = ckpt["args"]
    model = SignClassifier(FEATURES_PER_FRAME, a["num_classes"], a["arch"], a["hidden"],
                           a["num_layers"], a["dropout"])
    model.load_state_dict(ckpt["model"])
    return model.eval()


def load_labels(manifest: Path = LABEL_MANIFEST) -> list[str]:
    """label_id -> gloss, from the training manifest (the checkpoints don't store names)."""
    names: dict[int, str] = {}
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            names[int(row["label_id"])] = row["gloss"]
    return [names.get(i, f"class_{i}") for i in range(max(names) + 1)]


def missing_files(configs=MODEL_CONFIGS) -> list[Path]:
    """Checkpoint / manifest files the recognizer needs that are not on disk."""
    need = [LABEL_MANIFEST]
    for clf, ref in configs.values():
        need.append(PIPELINE_DIR / clf / "best.pt")
        if ref:
            need.append(PIPELINE_DIR / ref / "best.pt")
    return [p for p in need if not p.is_file()]


class Recognizer:
    """Stage 2 refiners + Stage 3 classifiers for every configuration in MODEL_CONFIGS."""

    def __init__(self, configs=MODEL_CONFIGS):
        self.labels = load_labels()
        self.configs = {}
        refiners: dict[str, TemporalRefiner] = {}
        for name, (clf_dir, ref_dir) in configs.items():
            clf = load_classifier(PIPELINE_DIR / clf_dir / "best.pt")
            ref = None
            if ref_dir:
                if ref_dir not in refiners:
                    refiners[ref_dir] = load_refiner(PIPELINE_DIR / ref_dir / "best.pt")
                ref = refiners[ref_dir]
            self.configs[name] = (clf, ref)

    # Stage 2 -------------------------------------------------------------- #
    @staticmethod
    @torch.no_grad()
    def refine(keypoints: np.ndarray, refiner: TemporalRefiner) -> np.ndarray:
        """(T, 73, 3) pixel keypoints in the 224 frame -> refined, same space."""
        seq = to_model_space(keypoints, FRAME_SIZE)
        out = refiner(torch.from_numpy(make_windows(seq)).to(CPU)).numpy()
        return to_pixel_space(out, FRAME_SIZE)

    # Stage 3 -------------------------------------------------------------- #
    @staticmethod
    @torch.no_grad()
    def classify(keypoints: np.ndarray, classifier: SignClassifier) -> np.ndarray:
        """(T, 73, 3) pixel keypoints -> class probabilities (num_classes,)."""
        feats = normalize_keypoints(keypoints, FRAME_SIZE).reshape(keypoints.shape[0], -1)
        logits = classifier(torch.from_numpy(feats)[None])
        return torch.softmax(logits, dim=-1)[0].numpy()

    def top_k(self, probs: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        order = np.argsort(probs)[::-1][:k]
        return [(self.labels[i], float(probs[i])) for i in order]

    def run(self, keypoints: np.ndarray) -> dict:
        """Run every configuration on one clip's Stage 1 keypoints (T, 73, 3)."""
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (73, 3):
            raise ValueError(f"expected (T, 73, 3) keypoints, got {keypoints.shape}")
        results = {}
        for name, (clf, ref) in self.configs.items():
            kps = self.refine(keypoints, ref) if ref is not None else keypoints
            probs = self.classify(kps, clf)
            results[name] = {"keypoints": kps, "probs": probs, "top5": self.top_k(probs)}
        return results


# --------------------------------------------------------------------------- #
# Frames -> 16 signer-centred 224x224 frames (same format as frames16/)
# --------------------------------------------------------------------------- #
def read_video_frames(path: str | Path, num_frames: int = NUM_FRAMES) -> list[np.ndarray]:
    """num_frames evenly spaced RGB frames from a video file (whole clip)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    try:
        if total > 0:
            wanted = even_indices(total, num_frames)
            unique = sorted(set(wanted.tolist()))
            got: dict[int, np.ndarray] = {}
            idx = 0
            while unique and idx <= unique[-1]:
                ok, bgr = cap.read()
                if not ok:
                    break
                if idx in unique:
                    got[idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                idx += 1
            if len(got) == len(unique):
                return [got[i] for i in wanted]
            # Frame count was wrong (common with webm/variable-rate files): fall back.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise ValueError("video contains no readable frames")
    return [frames[i] for i in even_indices(len(frames), num_frames)]


def stage1_keypoints(stage1, frames_rgb) -> np.ndarray:
    """Run Stage 1 on a list/array of RGB frames. stage1 = (model, args, device, predict_clip)."""
    model, args, device, predict_clip = stage1
    if len({f.shape for f in frames_rgb}) == 1:
        with torch.no_grad():
            return predict_clip(model, np.stack(list(frames_rgb)), args["input_h"],
                                args["input_w"], device).astype(np.float32)
    # Mixed frame sizes: run frame by frame.
    out = []
    with torch.no_grad():
        for f in frames_rgb:
            out.append(predict_clip(model, f[None], args["input_h"], args["input_w"], device)[0])
    return np.stack(out).astype(np.float32)


def signer_box_from_keypoints(keypoints: np.ndarray, frame_w: int, frame_h: int,
                              min_score: float = 0.3) -> list[float] | None:
    """Approximate a WLASL-style person box from full-frame Stage 1 keypoints.

    WLASL boxes come from a person detector and cover the head, torso and arms.
    Take the union of confident upper-body and hand points over all frames and
    pad it (more above for the head, a little to the sides for the hands).
    """
    pts = keypoints[keypoints[..., 2] >= min_score][:, :2]
    if len(pts) < 8:
        return None
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1, x2 = x1 - 0.20 * w, x2 + 0.20 * w
    y1, y2 = y1 - 0.30 * h, y2 + 0.20 * h
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(frame_w), x2), min(float(frame_h), y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def prepare_clip(frames_rgb: list[np.ndarray], crop: str = "auto", stage1=None,
                 bbox=None) -> tuple[np.ndarray, list[float] | None]:
    """16 RGB frames (any size) -> (16, 224, 224, 3) uint8 clip and the box used.

    crop: "auto"      signer box estimated with a first Stage 1 pass (needs stage1)
          "bbox"      use the given bbox (x1, y1, x2, y2) in frame pixels
          "letterbox" pad the whole frame to a square
    """
    if len(frames_rgb) != NUM_FRAMES:
        frames_rgb = [frames_rgb[i] for i in even_indices(len(frames_rgb), NUM_FRAMES)]
    h, w = frames_rgb[0].shape[:2]
    box = None
    if crop == "auto":
        if stage1 is None:
            raise ValueError("crop='auto' needs the Stage 1 model")
        box = signer_box_from_keypoints(stage1_keypoints(stage1, frames_rgb), w, h)
    elif crop == "bbox":
        box = list(bbox) if bbox is not None else None
    if box is not None:
        clip = [square_crop(f, box, FRAME_SIZE) for f in frames_rgb]
    else:
        clip = [letterbox(f, FRAME_SIZE) for f in frames_rgb]
    return np.stack(clip).astype(np.uint8), box


def recognize_frames(frames_rgb, stage1, recognizer: Recognizer, crop: str = "auto",
                     bbox=None) -> dict:
    """Full pipeline on raw frames. Returns clip, Stage 1 keypoints and per-model results."""
    clip, box = prepare_clip(frames_rgb, crop=crop, stage1=stage1, bbox=bbox)
    raw = stage1_keypoints(stage1, clip)
    return {"clip": clip, "box": box, "stage1": raw, "models": recognizer.run(raw)}


# --------------------------------------------------------------------------- #
# Stored WLASL clips (no video needed): keypoints from the original GPU run
# --------------------------------------------------------------------------- #
def stored_clips(split: str = "test") -> list[dict]:
    with open(LABEL_MANIFEST, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == split]


def load_stored_keypoints(video_id: str, condition: str = "clean") -> np.ndarray:
    return np.load(STORED_KEYPOINTS / condition / f"{video_id}.npy").astype(np.float32)


# --------------------------------------------------------------------------- #
# Temporal analysis helpers
# --------------------------------------------------------------------------- #
def group_confidence(keypoints: np.ndarray) -> dict[str, np.ndarray]:
    """Mean Stage 1/2 confidence per frame for each keypoint group."""
    return {g: keypoints[:, s, 2].mean(axis=1) for g, s in GROUP_SLICES.items()}


def correction_per_frame(raw: np.ndarray, refined: np.ndarray) -> dict[str, np.ndarray]:
    """How far Stage 2 moved points each frame (pixels in the 224 frame)."""
    move = np.linalg.norm(refined[..., :2] - raw[..., :2], axis=-1)  # (T, 73)
    weak = raw[..., 2] < 0.2
    return {
        "all points": move.mean(axis=1),
        "low-confidence points (<0.2)": np.where(weak.any(axis=1),
                                                 (move * weak).sum(1) / np.maximum(weak.sum(1), 1), 0.0),
    }


def hand_motion(keypoints: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame centre of each hand (mean of its 21 points), weighted by confidence."""
    out = {}
    for g in ("left_hand", "right_hand"):
        pts = keypoints[:, GROUP_SLICES[g]]
        w = np.clip(pts[..., 2], 1e-6, None)
        out[g] = (pts[..., :2] * w[..., None]).sum(1) / w.sum(1)[:, None]
    return out


def draw_keypoints(frame_rgb: np.ndarray, keypoints: np.ndarray, min_score: float = 0.2,
                   scale: int = 2) -> np.ndarray:
    """Overlay 73 keypoints on a 224 frame (upscaled for display)."""
    img = cv2.resize(frame_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    colors = {"upper_body": (0, 200, 0), "mouth": (255, 165, 0),
              "left_hand": (60, 120, 255), "right_hand": (255, 60, 60)}  # RGB
    for g, s in GROUP_SLICES.items():
        for x, y, c in keypoints[s]:
            if c >= min_score:
                cv2.circle(img, (int(round(x * scale)), int(round(y * scale))), scale + 1,
                           colors[g], -1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# Experimental live word prediction (webcam)
# --------------------------------------------------------------------------- #
def to_crop_space(keypoints: np.ndarray, box, frame_w: int, frame_h: int,
                  size: int = FRAME_SIZE) -> np.ndarray:
    """Map full-frame keypoints into the square signer crop the models expect.

    Matches sample_frames.square_crop (box given) or letterbox (box None) without
    re-running Stage 1 on the cropped frames.
    """
    if box is None:
        side = float(max(frame_w, frame_h))
        left, top = -(side - frame_w) / 2.0, -(side - frame_h) / 2.0
    else:
        x1, y1, x2, y2 = box
        side = float(max(x2 - x1, y2 - y1))
        left, top = (x1 + x2) / 2.0 - side / 2.0, (y1 + y2) / 2.0 - side / 2.0
    out = np.asarray(keypoints, dtype=np.float32).copy()
    out[..., 0] = (out[..., 0] - left) / side * size
    out[..., 1] = (out[..., 1] - top) / side * size
    return out


def hands_active(keypoints: np.ndarray, min_score: float = 0.3, min_motion: float = 0.04) -> bool:
    """True when at least one hand is visible in most frames and actually moves.

    keypoints are in the 224 crop; min_motion is the hand centre's range of
    movement as a fraction of the crop side.
    """
    for g in ("left_hand", "right_hand"):
        pts = keypoints[:, GROUP_SLICES[g]]
        conf = pts[..., 2].mean(axis=1)
        good = conf >= min_score
        if good.mean() < 0.5:
            continue
        w = np.clip(pts[good][..., 2], 1e-6, None)
        centre = (pts[good][..., :2] * w[..., None]).sum(1) / w.sum(1)[:, None]
        if np.ptp(centre, axis=0).max() / FRAME_SIZE >= min_motion:
            return True
    return False


class LiveWordPredictor:
    """Rolling-window word prediction from live Stage 1 keypoints.

    Thread-safe: push() is called from the webcam callback thread, state() from
    either thread. Every update_every_s it takes the last window_s seconds of
    keypoints, maps them into crop space, samples 16 frames and runs Stage 2 +
    Stage 3 (a few ms on the CPU). Probabilities are smoothed over updates and a
    word is marked "stable" once the top word repeats stable_updates times.
    """

    def __init__(self, recognizer: Recognizer, model_name: str = PRIMARY_MODEL,
                 window_s: float = 3.0, update_every_s: float = 0.5, min_frames: int = 8,
                 stable_updates: int = 3, smoothing: float = 0.5):
        import threading
        from collections import deque

        self.labels = recognizer.labels
        self.classifier, self.refiner = recognizer.configs[model_name]
        self.window_s, self.update_every_s = window_s, update_every_s
        self.min_frames, self.smoothing = min_frames, smoothing
        self._lock = threading.Lock()
        self._buffer = deque()
        self._size = (1, 1)
        self._last_update = 0.0
        self._ema = None
        self._history = deque(maxlen=stable_updates)
        self._state = {"status": "collecting", "word": None, "prob": 0.0, "top3": [], "frames": 0}

    def configure(self, window_s: float) -> None:
        with self._lock:
            self.window_s = float(window_s)

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def push(self, t: float, keypoints73: np.ndarray, frame_w: int, frame_h: int) -> None:
        with self._lock:
            self._buffer.append((t, np.asarray(keypoints73, dtype=np.float32)))
            self._size = (frame_w, frame_h)
            while self._buffer and self._buffer[0][0] < t - self.window_s:
                self._buffer.popleft()
            due = t - self._last_update >= self.update_every_s
            if due:
                self._last_update = t
                snapshot = [k for _, k in self._buffer]
                size = self._size
        if due:
            self._update(snapshot, size)

    def _set(self, **state) -> None:
        with self._lock:
            self._state = state

    def _update(self, frames: list[np.ndarray], size: tuple[int, int]) -> None:
        n = len(frames)
        if n < self.min_frames:
            self._set(status="collecting", word=None, prob=0.0, top3=[], frames=n)
            return
        kps = np.stack(frames)
        box = signer_box_from_keypoints(kps, *size)
        crop = to_crop_space(kps, box, *size)
        if not hands_active(crop):
            with self._lock:
                self._ema = None
                self._history.clear()
            self._set(status="idle", word=None, prob=0.0, top3=[], frames=n)
            return
        seq = crop[even_indices(n, NUM_FRAMES)]
        if self.refiner is not None:
            seq = Recognizer.refine(seq, self.refiner)
        probs = Recognizer.classify(seq, self.classifier)
        with self._lock:
            a = self.smoothing
            self._ema = probs if self._ema is None else a * probs + (1 - a) * self._ema
            ema = self._ema
            order = np.argsort(ema)[::-1][:3]
            self._history.append(int(order[0]))
            stable = (len(self._history) == self._history.maxlen
                      and len(set(self._history)) == 1)
        top3 = [(self.labels[i], float(ema[i])) for i in order]
        self._set(status="stable" if stable else "updating", word=top3[0][0],
                  prob=top3[0][1], top3=top3, frames=n)


def draw_live_word(frame_bgr: np.ndarray, state: dict) -> np.ndarray:
    """Draw the live word prediction as a banner at the bottom of a BGR frame (in place)."""
    h, w = frame_bgr.shape[:2]
    bar = max(44, h // 8)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, h - bar), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame_bgr, 0.35, 0, dst=frame_bgr)
    font, y = cv2.FONT_HERSHEY_SIMPLEX, h - bar // 2 + 6
    status = state.get("status")
    if status in ("updating", "stable"):
        color = (80, 220, 80) if status == "stable" else (0, 210, 255)  # green / amber (BGR)
        main = f"{state['word'].upper()}  {state['prob']:.0%}"
        cv2.putText(frame_bgr, main, (12, y), font, 0.8, color, 2, cv2.LINE_AA)
        rest = "   ".join(f"{wd} {p:.0%}" for wd, p in state["top3"][1:])
        cv2.putText(frame_bgr, rest, (min(w - 10, 12 + 20 * len(main)), y), font, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
    else:
        msg = (f"Live word: collecting frames ({state.get('frames', 0)})" if status == "collecting"
               else "Live word: waiting for signing (raise and move a hand)")
        cv2.putText(frame_bgr, msg, (12, y), font, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return frame_bgr
