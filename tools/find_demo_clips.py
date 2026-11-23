"""Find the best WLASL test clips for a live demo of the Word Recognition tab.

Ranks the 161 stored test clips by how well they show the project's story:
  1. On clean keypoints the Proposed model gets the word right.
  2. Under a damaged condition (default: missing_severe) Proposed still has the
     right word (1st, or in its top-5) while Frame-only loses it.

Usage (repo root, .venv active):
    python tools/find_demo_clips.py
    python tools/find_demo_clips.py --condition occlusion_severe --top 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.asl_pipeline import (  # noqa: E402
    CONDITIONS, PRIMARY_MODEL, Recognizer, load_stored_keypoints, stored_clips,
)

FRAME_ONLY = "Frame-only (no Stage 2)"


def rank_of(probs: np.ndarray, label: int) -> int:
    """1-based rank of the true label (1 = top prediction)."""
    return int((probs > probs[label]).sum()) + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="missing_severe", choices=CONDITIONS[1:])
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rec = Recognizer()
    rows = []
    for clip in stored_clips("test"):
        y = int(clip["label_id"])
        clean = rec.run(load_stored_keypoints(clip["video_id"], "clean"))
        bad = rec.run(load_stored_keypoints(clip["video_id"], args.condition))
        rows.append({
            "word": clip["gloss"], "clip": clip["video_id"],
            "clean_prop": rank_of(clean[PRIMARY_MODEL]["probs"], y),
            "clean_fo": rank_of(clean[FRAME_ONLY]["probs"], y),
            "bad_prop": rank_of(bad[PRIMARY_MODEL]["probs"], y),
            "bad_fo": rank_of(bad[FRAME_ONLY]["probs"], y),
        })

    # Best demo: right on clean, survives damage with Proposed, lost by Frame-only.
    demo = [r for r in rows if r["clean_prop"] == 1 and r["bad_prop"] <= 5 and r["bad_fo"] > 5]
    demo.sort(key=lambda r: (r["bad_prop"], -r["bad_fo"], r["clean_fo"]))
    reliable = sorted((r for r in rows if r["clean_prop"] == 1 and r["clean_fo"] == 1),
                      key=lambda r: r["word"])

    def show(title, items):
        print(f"\n{title} ({len(items)} clips)")
        print(f"  {'word':<14}{'clip':<8}{'clean P/FO':>12}{args.condition + ' P/FO':>26}")
        for r in items[: args.top]:
            print(f"  {r['word']:<14}{r['clip']:<8}{r['clean_prop']:>6}/{r['clean_fo']:<5}"
                  f"{r['bad_prop']:>18}/{r['bad_fo']}")

    print("Numbers are the rank of the TRUE word (1 = predicted correctly). "
          "P = Proposed, FO = Frame-only.")
    show(f"Best robustness demos: correct on clean, Proposed keeps it under "
         f"{args.condition}, Frame-only loses it", demo)
    show("Reliable clean clips: Proposed and Frame-only both correct", reliable)
    return 0


if __name__ == "__main__":
    sys.exit(main())
