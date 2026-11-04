"""Mean +/- std over seeds for the robustness table.

Seed 0 is runs/pipeline/robustness.csv (all columns, raw-frame merged in); seeds 1.. are runs/seeds/seedN/
robustness.csv (keypoint columns) and robustness_raw_frame.csv (raw-frame column).
Std is the sample std (n-1). A seed missing a column is skipped for that column and the count is shown.
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

CONDITIONS = ["clean"] + [f"{d}_{s}" for d in ("occlusion", "blur", "missing") for s in ("mild", "moderate", "severe")]


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", default="runs/pipeline/robustness.csv")
    p.add_argument("--seeds_dir", default="runs/seeds")
    p.add_argument("--out", default="runs/seeds/robustness_mean_std")
    p.add_argument("--metric", default="top1", choices=["top1", "top5", "macro_f1"])
    args = p.parse_args()

    runs = {0: read(args.main)}
    for d in sorted(glob.glob(os.path.join(args.seeds_dir, "seed*"))):
        seed = int(re.search(r"seed(\d+)$", d).group(1))
        rows = []
        # robustness.csv (keypoint columns), robustness_raw_frame.csv, robustness_<baseline>.csv
        for path in sorted(glob.glob(os.path.join(d, "robustness*.csv"))):
            if "mean_std" not in os.path.basename(path):
                rows += read(path)
        if rows:
            runs.setdefault(seed, []).extend(rows)  # seed 0 already holds the main run

    # values[(config, condition)][seed] = metric
    values = defaultdict(dict)
    configs = []
    for seed, rows in runs.items():
        for r in rows:
            if r["config"] not in configs:
                configs.append(r["config"])
            values[(r["config"], r["condition"])][seed] = float(r[args.metric]) * 100

    def cell(vals):
        v = np.array(list(vals.values()))
        if len(v) == 0:
            return "n/a", None, None
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        return f"{v.mean():.1f} ± {sd:.1f}", v.mean(), sd

    lines = [f"{args.metric} (%) on the test split, mean ± std over seeds "
             f"({', '.join(f'{c}: n={len({s for (cf, cd), d in values.items() if cf == c for s in d})}' for c in configs)})", "",
             "| Condition | " + " | ".join(configs) + " |", "| --- | " + " | ".join("---" for _ in configs) + " |"]
    csv_rows = []
    for cond in CONDITIONS:
        cells = []
        for c in configs:
            text, m, sd = cell(values.get((c, cond), {}))
            cells.append(text)
            csv_rows.append({"condition": cond, "config": c, "mean": m, "std": sd, "n_seeds": len(values.get((c, cond), {}))})
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")

    # per-seed average over the 9 degraded conditions
    cells, avg_rows = [], []
    for c in configs:
        per_seed = {}
        for seed in runs:
            vs = [values[(c, cond)].get(seed) for cond in CONDITIONS[1:]]
            if all(v is not None for v in vs):
                per_seed[seed] = float(np.mean(vs))
        text, m, sd = cell(per_seed)
        cells.append(text)
        avg_rows.append({"condition": "avg_9_degraded", "config": c, "mean": m, "std": sd, "n_seeds": len(per_seed)})
    lines.append("| **avg of 9 degraded** | " + " | ".join(cells) + " |")
    csv_rows += avg_rows

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + "_" + args.metric + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out + "_" + args.metric + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "config", "mean", "std", "n_seeds"])
        w.writeheader(); w.writerows(csv_rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
