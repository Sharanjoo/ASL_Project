import numpy as np


def interpolate_low_confidence(keypoints, threshold=0.2):
    """Literature baseline for Stage 2 (keypoint reconstruction by interpolation, as in Roh et al., 2024):
    each keypoint's frames scoring below the threshold take x, y and score linearly interpolated over time
    from that keypoint's trusted frames; before the first or after the last trusted frame the nearest
    trusted value is held. Keypoints with no trusted frame are left unchanged. Nothing is learned.

    keypoints: (T, K, 3) as (x, y, score). Returns a new array of the same shape.
    """
    out = keypoints.copy()
    t = np.arange(len(out))
    for j in range(out.shape[1]):
        trusted = out[:, j, 2] >= threshold
        if trusted.all() or not trusted.any():
            continue
        for c in range(3):
            out[~trusted, j, c] = np.interp(t[~trusted], t[trusted], out[trusted, j, c])
    return out
