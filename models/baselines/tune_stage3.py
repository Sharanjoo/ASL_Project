"""Per-extractor Stage 3 tuning, chosen on the validation split only.

For each extractor: train a small grid of bi-GRU settings on its clean keypoints with 2 seeds, pick the setting with the
best mean validation top-1, retrain that setting with 5 seeds, and evaluate on all 10 conditions (test split).
Outputs runs/tuning/<extractor>/ and runs/seeds/seed<N>/robustness_<extractor>_tuned.csv.

  python models/baselines/tune_stage3.py [stage1 rtmw mediapipe]
"""
import itertools
import json
import os
import subprocess
import sys

PY = sys.executable
EXTRACTORS = {
    "stage1": ("Frame-only (tuned)", os.path.join("runs", "pipeline", "keypoints")),
    "rtmw": ("RTMW (tuned)", os.path.join("runs", "baselines", "rtmw", "keypoints")),
    "mediapipe": ("MediaPipe (tuned)", os.path.join("runs", "baselines", "mediapipe", "keypoints")),
}
GRID = [  # hidden, dropout, lr   (the first entry is the untuned default)
    (128, 0.3, 1e-3), (256, 0.3, 1e-3), (128, 0.5, 1e-3), (256, 0.5, 1e-3), (128, 0.3, 3e-4), (256, 0.3, 3e-4),
]
SEARCH_SEEDS = (0, 1)
FINAL_SEEDS = (0, 1, 2, 3, 4)


def train(kp_root, out_dir, hidden, dropout, lr, seed):
    metrics = os.path.join(out_dir, "final_metrics.json")
    if not os.path.isfile(metrics):  # the script crashes at interpreter exit on Windows after saving, so check the file
        subprocess.run([PY, os.path.join("models", "recognition", "train_recognition.py"), "--keypoints", os.path.join(kp_root, "clean", "manifest_keypoints.csv"),
                        "--out_dir", out_dir, "--epochs", "150", "--hidden", str(hidden), "--dropout", str(dropout), "--lr", str(lr), "--seed", str(seed)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(metrics, encoding="utf-8"))


def main():
    names = sys.argv[1:] or list(EXTRACTORS)
    for name in names:
        label, kp_root = EXTRACTORS[name]
        root = os.path.join("runs", "tuning", name)
        os.makedirs(root, exist_ok=True)
        scores = []
        for hidden, dropout, lr in GRID:
            vals = []
            for seed in SEARCH_SEEDS:
                m = train(kp_root, os.path.join(root, f"h{hidden}_d{dropout}_lr{lr}_s{seed}"), hidden, dropout, lr, seed)
                vals.append(m["val"]["top1"])
            scores.append({"hidden": hidden, "dropout": dropout, "lr": lr, "val_top1": sum(vals) / len(vals)})
            print(f"{name}: hidden {hidden} dropout {dropout} lr {lr}: val top-1 {scores[-1]['val_top1']:.3f}", flush=True)
        best = max(scores, key=lambda s: s["val_top1"])
        json.dump({"grid": scores, "best": best}, open(os.path.join(root, "search.json"), "w"), indent=1)
        print(f"{name}: best {best}", flush=True)
        for seed in FINAL_SEEDS:
            out_dir = os.path.join(root, f"best_s{seed}")
            train(kp_root, out_dir, best["hidden"], best["dropout"], best["lr"], seed)
            dst = os.path.join("runs", "seeds", f"seed{seed}")
            os.makedirs(dst, exist_ok=True)
            subprocess.run([PY, os.path.join("models", "evaluate_robustness.py"), "--keypoint_root", kp_root,
                            "--out", os.path.join(dst, f"robustness_{name}_tuned.csv"), "--config", f"{label}={os.path.join(out_dir, 'best.pt')}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"{name}: done", flush=True)


if __name__ == "__main__":
    main()
