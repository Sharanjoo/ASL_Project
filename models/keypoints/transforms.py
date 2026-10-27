import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def box_to_center_size(bbox, aspect, padding=1.25):
    x, y, w, h = bbox
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return center, np.array([w, h], dtype=np.float32) * padding


def _third_point(a, b):
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def get_affine(center, size, rotation_deg, out_w, out_h, inverse=False):
    rot = np.deg2rad(rotation_deg)
    sin, cos = np.sin(rot), np.cos(rot)
    up = np.array([0.0, -size[1] * 0.5], dtype=np.float32)
    src_dir = np.array([up[0] * cos - up[1] * sin, up[0] * sin + up[1] * cos], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _third_point(src[0], src[1])
    dst[0] = [out_w * 0.5, out_h * 0.5]
    dst[1] = dst[0] + np.array([0.0, -out_h * 0.5], dtype=np.float32)
    dst[2] = _third_point(dst[0], dst[1])

    if inverse:
        return cv2.getAffineTransform(dst, src)
    return cv2.getAffineTransform(src, dst)


def apply_affine(points, matrix):
    return points @ matrix[:, :2].T + matrix[:, 2]


def normalize_image(rgb):
    img = rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)


def make_heatmaps(points, valid, heatmap_w, heatmap_h, stride, sigma):
    xs = np.arange(heatmap_w, dtype=np.float32)[None, None, :]
    ys = np.arange(heatmap_h, dtype=np.float32)[None, :, None]
    mx = (points[:, 0] / stride)[:, None, None]
    my = (points[:, 1] / stride)[:, None, None]
    heatmaps = np.exp(-((xs - mx) ** 2 + (ys - my) ** 2) / (2 * sigma ** 2))

    inside = (mx[:, 0, 0] >= 0) & (mx[:, 0, 0] < heatmap_w) & (my[:, 0, 0] >= 0) & (my[:, 0, 0] < heatmap_h)
    weight = (valid & inside).astype(np.float32)
    heatmaps *= weight[:, None, None]
    return heatmaps.astype(np.float32), weight


def decode_heatmaps(heatmaps, stride):
    n, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(n, k, -1)
    idx = flat.argmax(axis=-1)
    scores = flat.max(axis=-1)
    x = (idx % w).astype(np.float32)
    y = (idx // w).astype(np.float32)

    # Quarter-pixel shift toward the higher neighbour, as in SimpleBaseline/HRNet.
    xi = np.clip(x.astype(int), 1, w - 2)
    yi = np.clip(y.astype(int), 1, h - 2)
    ni, ki = np.meshgrid(np.arange(n), np.arange(k), indexing="ij")
    dx = heatmaps[ni, ki, yi, xi + 1] - heatmaps[ni, ki, yi, xi - 1]
    dy = heatmaps[ni, ki, yi + 1, xi] - heatmaps[ni, ki, yi - 1, xi]
    interior_x = (x >= 1) & (x <= w - 2)
    interior_y = (y >= 1) & (y <= h - 2)
    x += 0.25 * np.sign(dx) * interior_x
    y += 0.25 * np.sign(dy) * interior_y

    coords = np.stack([x, y], axis=-1) * stride
    return coords, scores
