import argparse
import json
import os
import random

import cv2
import numpy as np

GROUPS = {
    "body": ("keypoints", 17, (0, 255, 0)),
    "foot": ("foot_kpts", 6, (255, 255, 0)),
    "face": ("face_kpts", 68, (0, 165, 255)),
    "left_hand": ("lefthand_kpts", 21, (255, 0, 0)),
    "right_hand": ("righthand_kpts", 21, (0, 0, 255)),
}

BOX_GROUPS = {
    "face_box": (0, 165, 255),
    "lefthand_box": (255, 0, 0),
    "righthand_box": (0, 0, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay COCO-WholeBody keypoint annotations on their matching image."
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--ann")
    parser.add_argument("--img_dir")
    parser.add_argument("--image_id", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not args.demo and (not args.ann or not args.img_dir):
        parser.error("--ann and --img_dir are required unless --demo is set")

    return args


def parse_kpts(flat_list, n_points):
    arr = np.array(flat_list, dtype=float).reshape(n_points, 3)
    return arr


def draw_group(img, flat_list, n_points, color):
    if not flat_list:
        return

    points = parse_kpts(flat_list, n_points)
    for x, y, v in points:
        if v > 0:
            cv2.circle(img, (int(round(x)), int(round(y))), 2, color, -1)


def draw_box(img, box, color):
    if not box:
        return

    x, y, w, h = box
    if w <= 0 or h <= 0:
        return

    top_left = (int(round(x)), int(round(y)))
    bottom_right = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(img, top_left, bottom_right, color, 1)


def overlay_annotation(img, annotation_dict):
    for field_name, n_points, color in GROUPS.values():
        draw_group(img, annotation_dict.get(field_name), n_points, color)

    for box_field, color in BOX_GROUPS.items():
        draw_box(img, annotation_dict.get(box_field), color)


def make_demo_image_and_annotation():
    img = np.full((480, 640, 3), 128, dtype=np.uint8)

    def jitter_points(cx, cy, n_points, spread):
        flat = []
        for _ in range(n_points):
            x = cx + random.uniform(-spread, spread)
            y = cy + random.uniform(-spread, spread)
            v = 2
            flat.extend([x, y, v])
        return flat

    body = jitter_points(320, 220, 17, 80)
    foot = jitter_points(320, 420, 6, 30)
    face = jitter_points(320, 120, 68, 40)
    left_hand = jitter_points(220, 260, 21, 25)
    right_hand = jitter_points(420, 260, 21, 25)

    annotation = {
        "keypoints": body,
        "foot_kpts": foot,
        "face_kpts": face,
        "lefthand_kpts": left_hand,
        "righthand_kpts": right_hand,
        "face_box": [280, 80, 80, 80],
        "lefthand_box": [195, 235, 50, 50],
        "righthand_box": [395, 235, 50, 50],
    }

    return img, annotation


def run_demo(out_path):
    img, annotation = make_demo_image_and_annotation()
    overlay_annotation(img, annotation)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)

    print("Demo overlay written to", out_path)
    print("Groups drawn:")
    for name, (field_name, n_points, _color) in GROUPS.items():
        print(f"  {name}: {n_points} points ({field_name})")
    print("Boxes drawn:", ", ".join(BOX_GROUPS.keys()))


def run_real(ann_path, img_dir, out_path, image_id=None):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_filename = {image["id"]: image["file_name"] for image in data["images"]}

    annotations = data["annotations"]
    if image_id is not None:
        matching = [a for a in annotations if a["image_id"] == image_id]
        if not matching:
            raise ValueError(f"No annotation found for image_id {image_id}")
        annotation = matching[0]
    else:
        annotation = annotations[0]

    file_name = id_to_filename.get(annotation["image_id"])
    if file_name is None:
        raise ValueError(f"No image entry found for image_id {annotation['image_id']}")

    img_path = os.path.join(img_dir, file_name)
    img = cv2.imread(img_path)

    if img is None:
        raise RuntimeError(
            f"Could not read image at {img_path}. "
            "This usually means --img_dir does not match the split (train2017/val2017) "
            "that this annotation file was built from."
        )

    overlay_annotation(img, annotation)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, img)

    print("Overlay written to", out_path)
    print("Image:", img_path)
    print("Image id:", annotation["image_id"])


def main():
    args = parse_args()

    if args.demo:
        run_demo(args.out)
    else:
        run_real(args.ann, args.img_dir, args.out, args.image_id)


if __name__ == "__main__":
    main()
