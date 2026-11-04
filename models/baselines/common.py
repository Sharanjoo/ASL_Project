"""Third-party keypoint estimators mapped onto this project's 73-point layout (models/keypoints/keypoint_spec.py).

Both estimators return (num_frames, 73, 3) float32 arrays of (x, y, score) in the pixel space of the frames they are
given, exactly like predict_wlasl.py does for Stage 1. Points the estimator did not detect get (0, 0, 0).

  mediapipe: Holistic landmarker (pose 33 + face mesh 478 + 2 x 21 hand landmarks)
  rtmw:      RTMW whole-body model (COCO-WholeBody 133 points, the same ordering Stage 1 trains on) via rtmlib
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MEDIAPIPE_MODEL = os.path.join(ROOT, "runs", "baselines", "weights", "holistic_landmarker.task")

NUM_KEYPOINTS = 73

# --- MediaPipe -> 73-point layout -------------------------------------------------------------------------------
# upper body: nose, L/R eye, L/R ear, L/R shoulder, L/R elbow, L/R wrist (COCO order, signer's left first)
MP_POSE = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
# 20 iBUG mouth points (48-67) -> face-mesh vertices
MP_MOUTH = [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,   # outer lip 48-59
            78, 81, 13, 311, 308, 402, 14, 178]                    # inner lip 60-67


def _fill(out, rows, landmarks, w, h, score=None):
    for dst, src in rows:
        lm = landmarks[src]
        s = 1.0 if score is None else score(lm)
        out[dst] = (lm.x * w, lm.y * h, s)


def mediapipe_to_73(result, w, h, swap_hands=False):
    out = np.zeros((NUM_KEYPOINTS, 3), np.float32)
    if result.pose_landmarks:
        vis = lambda lm: float(np.clip(lm.visibility if lm.visibility is not None else 1.0, 0, 1))  # noqa: E731
        _fill(out, [(i, s) for i, s in enumerate(MP_POSE)], result.pose_landmarks, w, h, vis)
    if result.face_landmarks:
        _fill(out, [(11 + i, s) for i, s in enumerate(MP_MOUTH)], result.face_landmarks, w, h)
    left, right = result.left_hand_landmarks, result.right_hand_landmarks
    if swap_hands:
        left, right = right, left
    if left:
        _fill(out, [(31 + i, i) for i in range(21)], left, w, h)
    if right:
        _fill(out, [(52 + i, i) for i in range(21)], right, w, h)
    return out


class MediaPipeEstimator:
    name = "mediapipe"

    def __init__(self, swap_hands=False, **_):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        self.swap_hands = swap_hands
        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MEDIAPIPE_MODEL),
            running_mode=vision.RunningMode.IMAGE,
        )
        self.model = vision.HolisticLandmarker.create_from_options(options)

    def __call__(self, frames_rgb):
        out = []
        for frame in frames_rgb:
            h, w = frame.shape[:2]
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            out.append(mediapipe_to_73(self.model.detect(image), w, h, self.swap_hands))
        return np.stack(out)


# --- RTMW (COCO-WholeBody 133) -> 73-point layout -----------------------------------------------------------------
def _wholebody_indices():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "models", "keypoints"))
    from keypoint_spec import WHOLEBODY_INDICES
    return WHOLEBODY_INDICES


class RTMWEstimator:
    """RTMW pose model run top-down with the WHOLE frame as the person box (frames are already signer-centred
    crops, which is also how Stage 1 sees them), so no detector is needed and a person is always 'found'."""

    name = "rtmw"

    def __init__(self, device="cpu", mode="balanced", **_):
        import onnxruntime

        if device == "cuda" and hasattr(onnxruntime, "preload_dlls"):
            onnxruntime.preload_dlls()  # pulls the CUDA/cuDNN DLLs from the pip nvidia-* wheels
        from rtmlib import Wholebody

        self.model = Wholebody(mode=mode, backend="onnxruntime", device=device, to_openpose=False)
        self.indices = _wholebody_indices()

    def __call__(self, frames_rgb):
        out = []
        for frame in frames_rgb:
            h, w = frame.shape[:2]
            bgr = np.ascontiguousarray(frame[..., ::-1])
            keypoints, scores = self.model.pose_model(bgr, bboxes=[[0, 0, w, h]])
            arr = np.zeros((NUM_KEYPOINTS, 3), np.float32)
            if len(keypoints):
                arr[:, :2] = keypoints[0][self.indices]
                arr[:, 2] = np.clip(scores[0][self.indices], 0.0, 1.0)
            out.append(arr)
        return np.stack(out)


ESTIMATORS = {"mediapipe": MediaPipeEstimator, "rtmw": RTMWEstimator}
