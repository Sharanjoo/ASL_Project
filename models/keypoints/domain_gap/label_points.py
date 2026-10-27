# The points hand-labeled on WLASL frames for the proposal's domain-gap check, and where
# each one sits in the 73-point model output (models/keypoints/keypoint_spec.py).
#
# Left/right for body and hands follow COCO: the SIGNER's left, which appears on the
# IMAGE RIGHT when they face the camera. Mouth corners are named by image side instead,
# matching the iBUG face layout (point 48 is on the image left).

LABEL_POINTS = [
    # name, group, model indices scored against it, hint shown in the labeling tool
    ("nose", "upper_body", [0], "Tip of the nose"),
    ("left_shoulder", "upper_body", [5], "Signer's LEFT shoulder (image right)"),
    ("right_shoulder", "upper_body", [6], "Signer's RIGHT shoulder (image left)"),
    ("left_elbow", "upper_body", [7], "Signer's LEFT elbow (image right)"),
    ("right_elbow", "upper_body", [8], "Signer's RIGHT elbow (image left)"),
    ("left_wrist", "left_hand", [9, 31], "Signer's LEFT wrist (image right)"),
    ("right_wrist", "right_hand", [10, 52], "Signer's RIGHT wrist (image left)"),
    ("mouth_corner_image_left", "mouth", [11], "Mouth corner on the IMAGE-LEFT side"),
    ("mouth_corner_image_right", "mouth", [17], "Mouth corner on the IMAGE-RIGHT side"),
    ("upper_lip_top", "mouth", [14], "Top edge of the upper lip, centre"),
    ("lower_lip_bottom", "mouth", [20], "Bottom edge of the lower lip, centre"),
    ("left_thumb_tip", "left_hand", [35], "LEFT hand: thumb tip"),
    ("left_index_tip", "left_hand", [39], "LEFT hand: index fingertip"),
    ("left_middle_tip", "left_hand", [43], "LEFT hand: middle fingertip"),
    ("left_pinky_tip", "left_hand", [51], "LEFT hand: little fingertip"),
    ("right_thumb_tip", "right_hand", [56], "RIGHT hand: thumb tip"),
    ("right_index_tip", "right_hand", [60], "RIGHT hand: index fingertip"),
    ("right_middle_tip", "right_hand", [64], "RIGHT hand: middle fingertip"),
    ("right_pinky_tip", "right_hand", [72], "RIGHT hand: little fingertip"),
]

GROUPS = ("upper_body", "mouth", "left_hand", "right_hand")
