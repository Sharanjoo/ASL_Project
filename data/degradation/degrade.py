import argparse
import zlib

import cv2
import numpy as np

DEGRADATIONS = ("clean", "occlusion", "blur", "missing")
SEVERITIES = ("mild", "moderate", "severe")

# Fraction of frames hit, plus the per-type strength, for each severity.
SEVERITY_PARAMS = {
    "mild": {"frame_fraction": 0.25, "occluder_side": 0.20, "blur_kernel": 7},
    "moderate": {"frame_fraction": 0.50, "occluder_side": 0.25, "blur_kernel": 13},
    "severe": {"frame_fraction": 0.75, "occluder_side": 0.30, "blur_kernel": 21},
}

# Wrist indices in the 73-point subset (models/keypoints/keypoint_spec.py).
LEFT_WRIST = 9
RIGHT_WRIST = 10


def clip_rng(video_id, degradation, severity, seed=0):
    key = f"{video_id}|{degradation}|{severity}|{seed}".encode("utf-8")
    return np.random.default_rng(zlib.crc32(key))


def pick_frames(rng, num_frames, fraction):
    count = max(1, int(round(num_frames * fraction)))
    count = min(count, num_frames - 1)
    return np.sort(rng.choice(num_frames, size=count, replace=False))


def hand_center(frame_idx, rng, size, hand_keypoints, hand, score_threshold=0.3):
    if hand_keypoints is not None:
        x, y, score = hand_keypoints[frame_idx, LEFT_WRIST if hand == 0 else RIGHT_WRIST]
        if score >= score_threshold:
            return float(x), float(y)
    # Fallback: signing hands mostly sit in front of the torso, left or right of centre.
    # Positions measured off bbox-cropped WLASL frames (data/wlasl/sample_frames.py --crop bbox).
    cx = size * (0.32 if hand == 0 else 0.68) + rng.normal(0, 0.05 * size)
    cy = size * 0.72 + rng.normal(0, 0.05 * size)
    return cx, cy


def occlude(frame, center, side):
    out = frame.copy()
    h, w = frame.shape[:2]
    half = side / 2.0
    x0, y0 = int(max(center[0] - half, 0)), int(max(center[1] - half, 0))
    x1, y1 = int(min(center[0] + half, w)), int(min(center[1] + half, h))
    out[y0:y1, x0:x1] = 128
    return out


def motion_blur(frame, kernel_size, angle_deg):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rotation = cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(frame, -1, kernel)


def degrade_clip(frames, degradation, severity, rng, hand_keypoints=None):
    """frames: (T, H, W, 3) uint8. Returns degraded frames and a (T,) bool mask of frames hit."""
    num_frames, size = frames.shape[0], frames.shape[1]
    mask = np.zeros(num_frames, dtype=bool)
    if degradation == "clean":
        return frames, mask

    params = SEVERITY_PARAMS[severity]
    hit = pick_frames(rng, num_frames, params["frame_fraction"])
    mask[hit] = True
    out = frames.copy()

    if degradation == "occlusion":
        hand = int(rng.integers(2))
        side = params["occluder_side"] * size
        for t in hit:
            out[t] = occlude(frames[t], hand_center(t, rng, size, hand_keypoints, hand), side)
    elif degradation == "blur":
        for t in hit:
            out[t] = motion_blur(frames[t], params["blur_kernel"], rng.uniform(0, 180))
    elif degradation == "missing":
        out[hit] = 0
    else:
        raise ValueError(f"Unknown degradation: {degradation}")

    return out, mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview every degradation/severity on one sampled WLASL clip."
    )
    parser.add_argument("--frames", required=True, help="One clip's .npy from sample_frames.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", type=int, default=3, help="Show every Nth frame")
    return parser.parse_args()


def main():
    args = parse_args()
    frames = np.load(args.frames)
    video_id = args.frames.replace("\\", "/").split("/")[-1].split(".")[0]

    rows = [frames[:: args.every]]
    labels = ["clean"]
    for degradation in DEGRADATIONS[1:]:
        for severity in SEVERITIES:
            rng = clip_rng(video_id, degradation, severity)
            degraded, _ = degrade_clip(frames, degradation, severity, rng)
            rows.append(degraded[:: args.every])
            labels.append(f"{degradation}/{severity}")

    grid = []
    for label, row in zip(labels, rows):
        strip = cv2.cvtColor(np.hstack(list(row)), cv2.COLOR_RGB2BGR)
        cv2.putText(strip, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        grid.append(strip)
    cv2.imwrite(args.out, np.vstack(grid))

    print("Preview written to", args.out)
    print("Rows:", ", ".join(labels))


if __name__ == "__main__":
    main()
