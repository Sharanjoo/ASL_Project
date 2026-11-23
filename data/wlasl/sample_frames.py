"""Sample fixed-length, square, signer-centered frame clips from WLASL videos.

Reconstruction of the original data/wlasl/sample_frames.py, written from the
project README (the original was never committed). Behavior per the README:

  * --num_frames evenly spaced frames per clip (the later stages use 16).
  * --crop bbox (default): square crop around the WLASL signer bbox, padded with
    black where the square leaves the frame, so the signer is centered and
    undistorted. The bbox is stored at the source video's resolution; when a clip
    was downloaded smaller (e.g. 360p YouTube vs 720p metadata) the box is
    rescaled by how far it overshoots the frame.
  * --crop letterbox pads the whole frame to a square instead.
  * --crop stretch resizes the whole frame straight to a square.
  * Output: one (T, S, S, 3) uint8 RGB array per clip at <out_dir>/<video_id>.npy,
    plus <out_dir>/manifest_frames.csv = input columns + frames_path.
  * Prints OK / missing-video / bad-clip counts at the end.

Use verify_sampler.py (tools/) to compare this against the stored Stage 1
keypoints from the original run before trusting it.

The app can import sample_video() directly; with bbox=None it falls back to a
letterbox crop (used for webcam / user uploads that have no WLASL bbox).
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".swf")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def parse_bbox(value) -> list[float] | None:
    """Parse a WLASL bbox ("[x1, y1, x2, y2]" string or list). None if absent."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        box = [float(v) for v in value]
    else:
        text = str(value).strip()
        if not text or text.lower() in ("nan", "none"):
            return None
        box = [float(v) for v in ast.literal_eval(text)]
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def fit_bbox_to_frame(box: list[float], frame_w: int, frame_h: int) -> list[float]:
    """Rescale a source-resolution bbox if it overshoots the downloaded frame.

    If the clip is smaller than the metadata resolution, x2/y2 land outside the
    frame; scale the whole box down by the larger overshoot.
    """
    x1, y1, x2, y2 = box
    scale = min(1.0, frame_w / x2, frame_h / y2)
    return [x1 * scale, y1 * scale, x2 * scale, y2 * scale]


def square_crop(frame: np.ndarray, box: list[float], out_size: int) -> np.ndarray:
    """Square crop centered on box (side = longer box side), black padding, resize."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    side_i = max(1, int(round(side)))

    canvas = np.zeros((side_i, side_i, 3), dtype=frame.dtype)
    src_x1, src_y1 = max(0, left), max(0, top)
    src_x2, src_y2 = min(w, left + side_i), min(h, top + side_i)
    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1, dst_y1 = src_x1 - left, src_y1 - top
        canvas[dst_y1:dst_y1 + (src_y2 - src_y1), dst_x1:dst_x1 + (src_x2 - src_x1)] = \
            frame[src_y1:src_y2, src_x1:src_x2]
    return resize(canvas, out_size)


def letterbox(frame: np.ndarray, out_size: int) -> np.ndarray:
    """Pad the whole frame to a centered square with black, then resize."""
    h, w = frame.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=frame.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = frame
    return resize(canvas, out_size)


def resize(img: np.ndarray, out_size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if img.shape[0] > out_size else cv2.INTER_LINEAR
    return cv2.resize(img, (out_size, out_size), interpolation=interp)


# --------------------------------------------------------------------------- #
# Video reading
# --------------------------------------------------------------------------- #
def read_frame_range(path: str | os.PathLike, start: int, end: int | None) -> list[np.ndarray]:
    """Read frames [start, end] (0-based, inclusive; end=None -> last) as RGB."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    frames: list[np.ndarray] = []
    idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx >= start and (end is None or idx <= end):
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if end is not None and idx >= end:
                break
            idx += 1
    finally:
        cap.release()
    return frames


def even_indices(n_available: int, num_frames: int) -> np.ndarray:
    """num_frames evenly spaced indices over [0, n_available-1] (repeats if short)."""
    return np.linspace(0, n_available - 1, num_frames).round().astype(int)


def sample_video(
    path: str | os.PathLike,
    num_frames: int = 16,
    img_size: int = 224,
    crop: str = "bbox",
    bbox=None,
    frame_start: int = 1,
    frame_end: int = -1,
) -> np.ndarray | None:
    """Return a (num_frames, img_size, img_size, 3) uint8 RGB clip, or None.

    frame_start / frame_end follow WLASL conventions (1-based, -1 = last frame).
    If the file has fewer frames than frame_start, it is assumed to be an
    already-trimmed per-instance clip and all its frames are used.
    """
    start = max(0, int(frame_start) - 1)
    end = None if int(frame_end) < 0 else int(frame_end) - 1
    frames = read_frame_range(path, start, end)
    if not frames and start > 0:
        frames = read_frame_range(path, 0, None)  # already-trimmed clip
    if not frames:
        return None

    picked = [frames[i] for i in even_indices(len(frames), num_frames)]
    h, w = picked[0].shape[:2]
    box = parse_bbox(bbox)

    if crop == "bbox" and box is not None:
        box = fit_bbox_to_frame(box, w, h)
        out = [square_crop(f, box, img_size) for f in picked]
    elif crop == "stretch":
        out = [cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_AREA)
               for f in picked]
    else:  # letterbox, or bbox requested but none available
        out = [letterbox(f, img_size) for f in picked]
    return np.stack(out).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Manifest driver
# --------------------------------------------------------------------------- #
def youtube_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url or "")
    return m.group(1) if m else None


def find_video(video_dir: Path, row: dict) -> Path | None:
    """Locate a clip: <video_id>.<ext> (per-instance file) or <youtube_id>.<ext> (raw download)."""
    names = [row["video_id"], youtube_id(row.get("url", ""))]
    for name in filter(None, names):
        for ext in VIDEO_EXTS:
            p = video_dir / f"{name}{ext}"
            if p.exists():
                return p
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True, help="subset manifest.csv (or manifest_keypoints.csv)")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--crop", choices=("bbox", "letterbox", "stretch"), default="bbox")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N rows (0 = all)")
    args = ap.parse_args(argv)

    video_dir, out_dir = Path(args.video_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        in_fields = [f for f in reader.fieldnames if f not in ("frames_path", "keypoints_path")]
    if args.limit:
        rows = rows[: args.limit]

    ok_rows, n_missing, n_bad = [], 0, 0
    for i, row in enumerate(rows, 1):
        path = find_video(video_dir, row)
        if path is None:
            n_missing += 1
            continue
        # A per-instance file may already be trimmed; sample_video then uses the whole file.
        clip = sample_video(
            path, args.num_frames, args.img_size, args.crop, row.get("bbox"),
            int(row.get("frame_start", 1) or 1), int(row.get("frame_end", -1) or -1),
        )
        if clip is None:
            n_bad += 1
            print(f"[bad] {row['video_id']}: could not decode {path.name}", file=sys.stderr)
            continue
        out_path = out_dir / f"{row['video_id']}.npy"
        np.save(out_path, clip)
        out_row = {k: row.get(k, "") for k in in_fields}
        out_row["frames_path"] = out_path.as_posix()
        ok_rows.append(out_row)
        if i % 50 == 0:
            print(f"{i}/{len(rows)} processed")

    manifest_out = out_dir / "manifest_frames.csv"
    with open(manifest_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=in_fields + ["frames_path"])
        writer.writeheader()
        writer.writerows(ok_rows)

    print(f"OK: {len(ok_rows)}  missing video: {n_missing}  bad clip: {n_bad}")
    print(f"wrote {manifest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
