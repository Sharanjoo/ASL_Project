import json
import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from keypoint_spec import FLIP_PAIRS, subset_keypoints
from transforms import box_to_center_size, get_affine, apply_affine, make_heatmaps, normalize_image


def load_samples(ann_path, img_dir, min_visible=8, min_box_side=48):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_filename = {image["id"]: image["file_name"] for image in data["images"]}

    samples = []
    for annotation in data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue

        x, y, w, h = annotation["bbox"]
        if min(w, h) < min_box_side:
            continue

        kpts = subset_keypoints(annotation)
        if (kpts[:, 2] > 0).sum() < min_visible:
            continue

        samples.append(
            {
                "image_id": annotation["image_id"],
                "img_path": os.path.join(img_dir, id_to_filename[annotation["image_id"]]),
                "bbox": np.array(annotation["bbox"], dtype=np.float32),
                "kpts": kpts,
            }
        )
    return samples


def split_by_image_id(samples, holdout_mod=10):
    train = [s for s in samples if s["image_id"] % holdout_mod != 0]
    holdout = [s for s in samples if s["image_id"] % holdout_mod == 0]
    return train, holdout


class CocoWholeBodyKeypoints(Dataset):
    def __init__(self, samples, input_h=256, input_w=192, stride=4, sigma=2.0, train=False):
        self.samples = samples
        self.input_h = input_h
        self.input_w = input_w
        self.stride = stride
        self.sigma = sigma
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["img_path"])
        if img is None:
            raise RuntimeError(f"Could not read image at {sample['img_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        center, size = box_to_center_size(sample["bbox"], self.input_w / self.input_h)
        rotation = 0.0
        if self.train:
            size = size * random.uniform(0.75, 1.25)
            if random.random() < 0.6:
                rotation = random.uniform(-30, 30)

        matrix = get_affine(center, size, rotation, self.input_w, self.input_h)
        crop = cv2.warpAffine(img, matrix, (self.input_w, self.input_h), flags=cv2.INTER_LINEAR)

        points = apply_affine(sample["kpts"][:, :2], matrix)
        valid = sample["kpts"][:, 2] > 0

        if self.train and random.random() < 0.5:
            crop = crop[:, ::-1]
            points[:, 0] = self.input_w - 1 - points[:, 0]
            for a, b in FLIP_PAIRS:
                points[[a, b]] = points[[b, a]]
                valid[[a, b]] = valid[[b, a]]

        heatmaps, weight = make_heatmaps(
            points,
            valid,
            self.input_w // self.stride,
            self.input_h // self.stride,
            self.stride,
            self.sigma,
        )

        return {
            "image": normalize_image(np.ascontiguousarray(crop)),
            "heatmaps": heatmaps,
            "weight": weight,
            "center": center,
            "size": size,
            "kpts": sample["kpts"],
            "bbox": sample["bbox"],
        }
