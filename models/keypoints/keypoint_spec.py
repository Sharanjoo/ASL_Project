import numpy as np

# COCO-WholeBody 133-point order: body (17), foot (6), face (68), left hand (21), right hand (21).
FACE_OFFSET = 23
LEFT_HAND_OFFSET = 91
RIGHT_HAND_OFFSET = 112

# Signing-relevant subset: upper body (nose, eyes, ears, shoulders, elbows, wrists),
# the 20 iBUG mouth points (face 48-67), and both full hands.
UPPER_BODY = list(range(0, 11))
MOUTH = [FACE_OFFSET + i for i in range(48, 68)]
LEFT_HAND = [LEFT_HAND_OFFSET + i for i in range(21)]
RIGHT_HAND = [RIGHT_HAND_OFFSET + i for i in range(21)]

WHOLEBODY_INDICES = UPPER_BODY + MOUTH + LEFT_HAND + RIGHT_HAND
NUM_KEYPOINTS = len(WHOLEBODY_INDICES)

GROUP_SLICES = {
    "upper_body": slice(0, 11),
    "mouth": slice(11, 31),
    "left_hand": slice(31, 52),
    "right_hand": slice(52, 73),
}

# BGR, matching data/coco_wholebody/explore_annotations.py
GROUP_COLORS = {
    "upper_body": (0, 255, 0),
    "mouth": (0, 165, 255),
    "left_hand": (255, 0, 0),
    "right_hand": (0, 0, 255),
}


def build_flip_pairs():
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

    mouth_start = GROUP_SLICES["mouth"].start
    mouth_pairs = [(0, 6), (1, 5), (2, 4), (7, 11), (8, 10), (12, 16), (13, 15), (17, 19)]
    pairs += [(mouth_start + a, mouth_start + b) for a, b in mouth_pairs]

    left_start = GROUP_SLICES["left_hand"].start
    right_start = GROUP_SLICES["right_hand"].start
    pairs += [(left_start + i, right_start + i) for i in range(21)]
    return pairs


FLIP_PAIRS = build_flip_pairs()


def wholebody_array(annotation):
    flat = (
        annotation["keypoints"]
        + annotation["foot_kpts"]
        + annotation["face_kpts"]
        + annotation["lefthand_kpts"]
        + annotation["righthand_kpts"]
    )
    return np.array(flat, dtype=np.float32).reshape(133, 3)


def subset_keypoints(annotation):
    return wholebody_array(annotation)[WHOLEBODY_INDICES]
