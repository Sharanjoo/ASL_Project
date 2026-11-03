"""Score the results folder that run_npu.py produced on the NPU lab machine, against the FP32 runs.

    python models/npu/import_npu_results.py --results ../results_npu --out runs/pipeline_npu

  1. WLASL: builds a keypoint manifest per condition from the NPU keypoints and re-runs the robustness
     table with the Stage 2/3 models trained on FP32 keypoints (no retraining), next to the FP32 table.
  2. Domain gap: scores the NPU keypoints on the 100 hand-labeled WLASL frames.
  3. COCO and speed: copies run_npu.py's numbers into a summary.
Writes <out>/summary.md and prints it.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def build_keypoint_root(results, fp32_root, out_root, splits):
    conditions = sorted(os.listdir(os.path.join(results, "wlasl")))
    for condition in conditions:
        rows, fields = read_csv(os.path.join(fp32_root, condition, "manifest_keypoints.csv"))
        out_dir = os.path.join(out_root, condition)
        os.makedirs(out_dir, exist_ok=True)
        kept = []
        for row in rows:
            source = os.path.join(results, "wlasl", condition, f"{row['video_id']}.npy")
            if row["split"] not in splits:
                continue
            assert os.path.isfile(source), f"NPU results are missing {condition}/{row['video_id']}.npy"
            target = os.path.join(out_dir, f"{row['video_id']}.npy")
            shutil.copy(source, target)
            kept.append({**row, "keypoints_path": target})
        with open(os.path.join(out_dir, "manifest_keypoints.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
        print(f"  {condition:20s} {len(kept)} clips ({' '.join(splits)})")
    return conditions


def top1_table(path):
    rows, _ = read_csv(path)
    return {(r["condition"], r["config"]): float(r["top1"]) for r in rows}


def main():
    parser = argparse.ArgumentParser(description="Score NPU keypoint results against the FP32 runs.")
    parser.add_argument("--results", required=True, help="results_npu folder copied back from the lab machine")
    parser.add_argument("--out", default="runs/pipeline_npu")
    parser.add_argument("--fp32_pipeline", default="runs/pipeline", help="run_pipeline.sh output of the FP32 run")
    parser.add_argument("--reference", help="Float ONNX run of the same bundle, for the FP32 COCO numbers")
    parser.add_argument("--cpu_speed", help="results_onnx folder timed on the NPU machine's own CPU")
    parser.add_argument("--fp32_domain_gap", default="runs/domain_gap_metrics.json")
    parser.add_argument("--label_set", default="domain_gap/label_set.npz")
    parser.add_argument("--labels", default="runs/domain_gap_labels_Sharan_corners_swapped.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--prepare_for_retraining", action="store_true",
                        help="Only build keypoint manifests for all splits, ready for SKIP_KEYPOINTS=1 run_pipeline.sh")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fp32 = args.fp32_pipeline

    print("Building NPU keypoint manifests")
    splits = ("train", "val", "test") if args.prepare_for_retraining else (args.split,)
    build_keypoint_root(args.results, os.path.join(fp32, "keypoints"), os.path.join(args.out, "keypoints"), splits)
    if args.prepare_for_retraining:
        print(f"Next: SKIP_KEYPOINTS=1 bash models/run_pipeline.sh unused {args.out}")
        return

    subprocess.run([
        sys.executable, os.path.join(REPO, "models", "evaluate_robustness.py"),
        "--keypoint_root", os.path.join(args.out, "keypoints"), "--split", args.split,
        "--out", os.path.join(args.out, "robustness.csv"),
        "--config", f"Frame-only={fp32}/classifier_frame_only/best.pt",
        "--config", f"+Interpolation={fp32}/classifier_frame_only/best.pt:interpolate",
        "--config", f"+Temporal={fp32}/classifier_refined_synthetic/best.pt:{fp32}/refiner_synthetic/best.pt",
        "--config", f"+Temporal & robust training={fp32}/classifier_refined_degraded/best.pt:{fp32}/refiner_degraded/best.pt",
    ], check=True)

    domain_gap = os.path.join(args.results, "domain_gap_keypoints.npy")
    if os.path.isfile(domain_gap):
        subprocess.run([
            sys.executable, os.path.join(REPO, "models", "keypoints", "domain_gap", "score_domain_gap.py"),
            "--predictions", domain_gap, "--label_set", args.label_set, "--labels", args.labels,
            "--out", os.path.join(args.out, "domain_gap_metrics.json"),
        ], check=True)

    def load_json(folder, name):
        path = os.path.join(folder, name) if folder else ""
        return json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}

    npu_coco, ref_coco = load_json(args.results, "coco_metrics.json"), load_json(args.reference, "coco_metrics.json")
    npu_speed, ref_speed = load_json(args.results, "speed.json"), load_json(args.cpu_speed, "speed.json")
    npu_info = load_json(args.results, "run_info.json")

    lines = ["# Stage 1 on the Mobilint NPU", "", f"NPU run: `{npu_info.get('model', '?')}`"
             + (f" with CPU head `{npu_info['head_onnx']}`" if npu_info.get("head_onnx") else ""), ""]
    if npu_coco:
        lines += ["## COCO-WholeBody val2017 evaluation subset", "",
                  "| Group | FP32 PCK@0.05 | NPU PCK@0.05 | FP32 PCK@0.1 | NPU PCK@0.1 |", "| --- | --- | --- | --- | --- |"]
        for group in ("upper_body", "mouth", "left_hand", "right_hand", "all"):
            f = lambda m, k: f"{m[k] * 100:.1f}%" if k in m else "-"  # noqa: E731
            lines.append(f"| {group} | {f(ref_coco, f'{group}_pck@0.05')} | {f(npu_coco, f'{group}_pck@0.05')} | "
                         f"{f(ref_coco, f'{group}_pck@0.1')} | {f(npu_coco, f'{group}_pck@0.1')} |")
        lines.append("")
    if npu_speed:
        lines += ["## Speed (ms per frame)", "", "| | Accelerator | End to end |", "| --- | --- | --- |",
                  f"| NPU | {npu_speed['accelerator_ms_per_frame']} | {npu_speed['end_to_end_ms_per_frame']} |"]
        if ref_speed:
            lines.append(f"| CPU, ONNX | {ref_speed['accelerator_ms_per_frame']} | {ref_speed['end_to_end_ms_per_frame']} |")
        lines.append("")

    npu_gap = load_json(args.out, "domain_gap_metrics.json").get("groups", {})
    if npu_gap:
        with open(args.fp32_domain_gap, encoding="utf-8") as f:
            fp32_gap = json.load(f)["groups"]
        lines += ["## WLASL domain-gap frames (fraction of shoulder width)", "",
                  "| Group | FP32 PCK@0.1 | NPU PCK@0.1 | FP32 PCK@0.2 | NPU PCK@0.2 |", "| --- | --- | --- | --- | --- |"]
        for group, m in npu_gap.items():
            lines.append(f"| {group} | {fp32_gap[group]['pck@0.1'] * 100:.1f}% | {m['pck@0.1'] * 100:.1f}% | "
                         f"{fp32_gap[group]['pck@0.2'] * 100:.1f}% | {m['pck@0.2'] * 100:.1f}% |")
        lines.append("")

    fp32_top1, npu_top1 = top1_table(os.path.join(fp32, "robustness.csv")), top1_table(os.path.join(args.out, "robustness.csv"))
    configs = list(dict.fromkeys(c for _, c in npu_top1))
    lines += ["## WLASL100 test top-1: FP32 keypoints vs NPU keypoints", "",
              "| Condition | " + " | ".join(f"{c} FP32 | {c} NPU" for c in configs) + " |",
              "| --- |" + " --- |" * (2 * len(configs))]
    for condition in dict.fromkeys(c for c, _ in npu_top1):
        cells = [f"{fp32_top1.get((condition, c), float('nan')) * 100:.1f}% | {npu_top1[(condition, c)] * 100:.1f}%"
                 for c in configs]
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")

    summary = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n" + summary)


if __name__ == "__main__":
    main()
