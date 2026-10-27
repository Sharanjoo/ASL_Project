import csv

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_KEYPOINTS = 73
FEATURES_PER_FRAME = NUM_KEYPOINTS * 3


def read_keypoint_manifest(path, split=None):
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    return rows


def normalize_keypoints(keypoints, frame_size=224):
    """(T, 73, 3) pixel coords + score -> coords centred on the frame in [-0.5, 0.5], score kept."""
    out = keypoints.astype(np.float32).copy()
    out[..., :2] = out[..., :2] / frame_size - 0.5
    return out


class KeypointSequenceDataset(Dataset):
    def __init__(self, rows, train=False, frame_size=224, arrays=None):
        self.rows = rows
        self.train = train
        self.frame_size = frame_size
        self.arrays = arrays  # optional {keypoints_path: array} used instead of reading from disk

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        raw = self.arrays[row["keypoints_path"]] if self.arrays is not None else np.load(row["keypoints_path"])
        keypoints = normalize_keypoints(raw, self.frame_size)

        if self.train:
            scale = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.05, 0.05, size=2)
            keypoints[..., :2] = keypoints[..., :2] * scale + shift
            keypoints[..., :2] += np.random.normal(0, 0.005, size=keypoints[..., :2].shape)

        features = torch.from_numpy(keypoints.reshape(keypoints.shape[0], -1))
        return features, int(row["label_id"])
