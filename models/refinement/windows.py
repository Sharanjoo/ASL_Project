import csv

import numpy as np

NUM_KEYPOINTS = 73
WINDOW_RADIUS = 2  # t-2 .. t+2, as in the proposal


def load_keypoint_dir(manifest_path, split=None):
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    return {row["video_id"]: np.load(row["keypoints_path"]) for row in rows}, rows


def to_model_space(keypoints, frame_size=224):
    out = keypoints.astype(np.float32).copy()
    out[..., :2] = out[..., :2] / frame_size - 0.5
    return out


def to_pixel_space(keypoints, frame_size=224):
    out = keypoints.astype(np.float32).copy()
    out[..., :2] = (out[..., :2] + 0.5) * frame_size
    return out


def make_windows(sequence, radius=WINDOW_RADIUS):
    """(T, K, 3) -> (T, 2*radius+1, K, 3), edge frames repeated at clip boundaries."""
    num_frames = sequence.shape[0]
    offsets = np.arange(-radius, radius + 1)
    index = np.clip(np.arange(num_frames)[:, None] + offsets[None, :], 0, num_frames - 1)
    return sequence[index]


def synthetic_corruption(sequence, rng, frame_drop=0.15, point_drop=0.2, noise=0.02):
    """Keypoint-space corruption for the no-robustness-training baseline (proposal Baseline B)."""
    out = sequence.copy()
    num_frames, num_points = out.shape[:2]

    dropped_points = rng.random((num_frames, num_points)) < point_drop
    out[dropped_points] = [0.0, 0.0, 0.0]

    dropped_frames = rng.random(num_frames) < frame_drop
    dropped_frames[rng.integers(num_frames)] = False
    out[dropped_frames] = [0.0, 0.0, 0.0]

    out[..., :2] += rng.normal(0, noise, size=out[..., :2].shape) * (out[..., 2:3] > 0)
    return out
