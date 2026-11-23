"""Check that the Stage 2/3 checkpoints load and reproduce the original results.

Runs every model configuration on the stored test-split keypoints for each
condition and compares top-1 / top-5 with runs/pipeline/robustness.csv from
the original GPU run. Matching numbers mean the checkpoints, label mapping,
normalisation and your environment are all correct.

Usage (from the repo root, with .venv active):
    python tools/check_stage23.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.asl_pipeline import (  # noqa: E402
    CONDITIONS, MODEL_CONFIGS, PIPELINE_DIR, Recognizer, load_stored_keypoints,
    missing_files, stored_clips,
)

# Column names used in the original robustness table
ORIGINAL_COLUMN = {
    "Proposed (+Temporal & robust training)": "+Temporal & robust training",
    "+Temporal (synthetic-noise refiner)": "+Temporal",
    "Frame-only (no Stage 2)": "Frame-only",
}


def read_original(path: Path) -> dict:
    """{(condition, config): (top1 %, top5 %)} from evaluate_robustness.py's CSV."""
    if not path.is_file():
        return {}
    with open(path, newline="") as fh:
        return {(r["condition"], r["config"]): (100 * float(r["top1"]), 100 * float(r["top5"]))
                for r in csv.DictReader(fh)}


def main() -> int:
    missing = missing_files()
    if missing:
        print("Missing files:\n  " + "\n  ".join(map(str, missing)))
        return 1

    t0 = time.time()
    rec = Recognizer()
    print(f"Loaded {len(rec.configs)} configurations, {len(rec.labels)} classes "
          f"in {time.time() - t0:.1f}s")

    clips = stored_clips("test")
    labels = [int(c["label_id"]) for c in clips]
    original = read_original(PIPELINE_DIR / "robustness.csv")
    print(f"Test clips: {len(clips)}\n")

    worst = 0.0
    header = f"{'condition':<20}" + "".join(f"{ORIGINAL_COLUMN[n]:>30}" for n in MODEL_CONFIGS)
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        hits = {name: [0, 0] for name in MODEL_CONFIGS}
        for clip, y in zip(clips, labels):
            results = rec.run(load_stored_keypoints(clip["video_id"], cond))
            for name in MODEL_CONFIGS:
                order = np.argsort(results[name]["probs"])[::-1]
                hits[name][0] += int(order[0] == y)
                hits[name][1] += int(y in order[:5])
        cells = []
        for name in MODEL_CONFIGS:
            t1, t5 = (100 * h / len(clips) for h in hits[name])
            ref = original.get((cond, ORIGINAL_COLUMN[name]))
            if ref:
                worst = max(worst, abs(t1 - ref[0]))
                cells.append(f"{t1:5.1f} ({t5:4.1f}) vs {ref[0]:5.1f}")
            else:
                cells.append(f"{t1:5.1f} ({t5:4.1f})")
        print(f"{cond:<20}" + "".join(f"{c:>30}" for c in cells))

    print(f"\nFormat: top-1 (top-5) here vs top-1 in the original run.  "
          f"Took {time.time() - t0:.1f}s")
    if original:
        verdict = "MATCHES the original run" if worst < 0.7 else \
                  f"DIFFERS from the original run (largest top-1 gap {worst:.1f} points)"
        print(f"Verdict: {verdict}")
    else:
        print("No runs/pipeline/robustness.csv found to compare against.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
