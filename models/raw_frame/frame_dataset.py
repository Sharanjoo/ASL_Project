import csv
import os
import sys

import cv2
import numpy as np
from torch.utils.data import Dataset

from frame_model import frames_to_tensor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "degradation"))
from degrade import clip_rng, degrade_clip  # noqa: E402


def read_frame_manifest(path, split=None):
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    return rows


def augment(frames, rng):
    """One shared random scale/shift for the whole clip. No flip: it would swap the signing hand."""
    size = frames.shape[1]
    scale = rng.uniform(0.9, 1.1)
    shift = rng.uniform(-0.05, 0.05, size=2) * size
    matrix = np.array([[scale, 0, shift[0] + (1 - scale) * size / 2],
                       [0, scale, shift[1] + (1 - scale) * size / 2]], dtype=np.float32)
    return np.stack([cv2.warpAffine(f, matrix, (size, size), flags=cv2.INTER_LINEAR) for f in frames])


class FrameClipDataset(Dataset):
    """Clips of sampled frames, optionally degraded on the fly exactly as predict_wlasl.py does.

    The degradation is seeded per video/type/severity by the same clip_rng, so these are the
    same frames the keypoint pipeline loses. Pass hand_keypoints_dir (the clean keypoint run) so
    occluders land on the same wrists.
    """

    def __init__(self, rows, train=False, degradation="clean", severity="moderate", hand_keypoints_dir=None, seed=0):
        self.rows = rows
        self.train = train
        self.degradation = degradation
        self.severity = severity
        self.hand_keypoints_dir = hand_keypoints_dir
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        frames = np.load(row["frames_path"])

        if self.degradation != "clean":
            hand_keypoints = None
            if self.hand_keypoints_dir:
                hand_path = os.path.join(self.hand_keypoints_dir, f"{row['video_id']}.npy")
                if os.path.isfile(hand_path):
                    hand_keypoints = np.load(hand_path)
            rng = clip_rng(row["video_id"], self.degradation, self.severity, self.seed)
            frames, _ = degrade_clip(frames, self.degradation, self.severity, rng, hand_keypoints)

        if self.train:
            frames = augment(frames, np.random.default_rng())

        return frames_to_tensor(frames), int(row["label_id"])
