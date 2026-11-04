"""Paired bootstrap confidence intervals over the 258 test clips.

For every model, per-clip correctness is computed on each of the 10 conditions for each of the 5 seeds, then averaged over
seeds (a clip's score is the share of seeds that classify it correctly). Differences between two models are bootstrapped
by resampling clips with replacement (the same clips for both models, so it is a paired test).

Run from the repo root:  python models/baselines/bootstrap_ci.py
Writes report/results/baselines/bootstrap_ci.{csv,md}
"""
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from evaluate_robustness import CONDITION_ORDER, load_classifier, load_refiner, refine_rows  # noqa: E402
from keypoint_sequences import KeypointSequenceDataset, read_keypoint_manifest  # noqa: E402
from metrics import predict_logits  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = range(5)
N_BOOT = 10000


def p_main(*parts):
    return os.path.join("runs", *parts)


def seed_dir(seed):
    return p_main("pipeline") if seed == 0 else p_main("seeds", f"seed{seed}")


MODELS = {
    "Frame-only (Stage 1)": lambda s: (os.path.join(seed_dir(s), "classifier_frame_only", "best.pt"), None, p_main("pipeline", "keypoints")),
    "Proposed (Stage 1 + refiner)": lambda s: (os.path.join(seed_dir(s), "classifier_refined_degraded", "best.pt"),
                                               os.path.join(seed_dir(s), "refiner_degraded", "best.pt"), p_main("pipeline", "keypoints")),
    "RTMW": lambda s: (p_main("baselines", "rtmw", f"classifier_seed{s}", "best.pt"), None, p_main("baselines", "rtmw", "keypoints")),
    "MediaPipe": lambda s: (p_main("baselines", "mediapipe", f"classifier_seed{s}", "best.pt"), None, p_main("baselines", "mediapipe", "keypoints")),
}
PAIRS = [
    ("Proposed (Stage 1 + refiner)", "Frame-only (Stage 1)"),
    ("RTMW", "Frame-only (Stage 1)"),
    ("RTMW", "Proposed (Stage 1 + refiner)"),
    ("Frame-only (Stage 1)", "MediaPipe"),
    ("RTMW", "MediaPipe"),
]
TARGETS = {"clean": ["clean"], "average of 9 degraded": CONDITION_ORDER[1:], "severe blur": ["blur_severe"],
           "severe occlusion": ["occlusion_severe"], "severe missing frames": ["missing_severe"]}


def correctness(name):
    """(conditions, clips) array of the share of seeds that classify each clip correctly."""
    acc = np.zeros((len(CONDITION_ORDER), 0))
    per_seed = []
    for seed in SEEDS:
        clf_path, ref_path, kp_root = MODELS[name](seed)
        clf, _ = load_classifier(clf_path, device)
        refiner = load_refiner(ref_path, device) if ref_path else None
        rows_by_cond = []
        for cond in CONDITION_ORDER:
            rows = read_keypoint_manifest(os.path.join(kp_root, cond, "manifest_keypoints.csv"), "test")
            arrays = refine_rows(rows, refiner, device) if refiner is not None else None
            logits, labels = predict_logits(clf, DataLoader(KeypointSequenceDataset(rows, arrays=arrays), batch_size=64), device)
            rows_by_cond.append((logits.argmax(1) == labels).astype(float))
        per_seed.append(np.stack(rows_by_cond))
    return np.mean(per_seed, axis=0)


def boot(diff, rng):
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    return diff.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), float((means <= 0).mean())


def main():
    rng = np.random.default_rng(0)
    acc = {}
    for name in MODELS:
        print("computing", name, flush=True)
        acc[name] = correctness(name)
    cidx = {c: i for i, c in enumerate(CONDITION_ORDER)}
    rows = []
    for target, conds in TARGETS.items():
        sel = [cidx[c] for c in conds]
        for name in MODELS:
            per_clip = acc[name][sel].mean(axis=0) * 100
            m, lo, hi, _ = boot(per_clip, rng)
            rows.append({"kind": "accuracy", "target": target, "a": name, "b": "", "mean": m, "lo": lo, "hi": hi, "p_le_0": ""})
        for a, b in PAIRS:
            d = (acc[a][sel].mean(axis=0) - acc[b][sel].mean(axis=0)) * 100
            m, lo, hi, p = boot(d, rng)
            rows.append({"kind": "difference", "target": target, "a": a, "b": b, "mean": m, "lo": lo, "hi": hi, "p_le_0": p})
    out = os.path.join("report", "results", "baselines")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "bootstrap_ci.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = ["# Paired bootstrap over the 258 test clips", "",
             f"{N_BOOT} resamples of clips, each clip scored as the share of 5 seeds that classify it correctly. "
             "95% intervals in percentage points. `p(diff <= 0)` is the share of resamples where A is not better than B.", "",
             "## Accuracy with 95% interval (%)", "", "| Target | " + " | ".join(MODELS) + " |", "| --- | " + " | ".join("---" for _ in MODELS) + " |"]
    for target in TARGETS:
        cells = []
        for name in MODELS:
            r = next(r for r in rows if r["kind"] == "accuracy" and r["target"] == target and r["a"] == name)
            cells.append(f"{r['mean']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}]")
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences A − B with 95% interval (points)", "", "| Target | A | B | A − B | 95% interval | p(diff <= 0) | Excludes 0? |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        if r["kind"] == "difference":
            ex = "yes" if (r["lo"] > 0 or r["hi"] < 0) else "no"
            lines.append(f"| {r['target']} | {r['a']} | {r['b']} | {r['mean']:+.1f} | [{r['lo']:+.1f}, {r['hi']:+.1f}] | {r['p_le_0']:.3f} | {ex} |")
    open(os.path.join(out, "bootstrap_ci.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", os.path.join(out, "bootstrap_ci.md"))


if __name__ == "__main__":
    main()
