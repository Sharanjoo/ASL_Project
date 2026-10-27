import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "recognition"))
sys.path.insert(0, os.path.join(HERE, "refinement"))

from classifier import SignClassifier  # noqa: E402
from keypoint_sequences import FEATURES_PER_FRAME, KeypointSequenceDataset, read_keypoint_manifest  # noqa: E402
from metrics import classification_metrics, predict_logits  # noqa: E402
from refiner import TemporalRefiner  # noqa: E402
from windows import make_windows, to_model_space, to_pixel_space  # noqa: E402

CONDITION_ORDER = ["clean"] + [
    f"{d}_{s}" for d in ("occlusion", "blur", "missing") for s in ("mild", "moderate", "severe")
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill the proposal's robustness table: every model configuration x every degradation condition."
    )
    parser.add_argument(
        "--keypoint_root", required=True,
        help="Folder with one sub-folder per condition (clean, occlusion_mild, ...) holding manifest_keypoints.csv",
    )
    parser.add_argument(
        "--config", action="append", required=True,
        help='"Name=classifier.pt" or "Name=classifier.pt:refiner.pt"; repeat once per column',
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, help="Output CSV path; a .md table is written next to it")
    return parser.parse_args()


def load_classifier(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    a = checkpoint["args"]
    model = SignClassifier(FEATURES_PER_FRAME, a["num_classes"], a["arch"], a["hidden"], a["num_layers"], a["dropout"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), a["num_classes"]


def load_refiner(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    a = checkpoint["args"]
    model = TemporalRefiner(arch=a["arch"], hidden=a["hidden"], num_layers=a["num_layers"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def refine_rows(rows, refiner, device):
    refined = {}
    with torch.no_grad():
        for row in rows:
            sequence = to_model_space(np.load(row["keypoints_path"]))
            out = refiner(torch.from_numpy(make_windows(sequence)).to(device)).cpu().numpy()
            refined[row["keypoints_path"]] = to_pixel_space(out)
    return refined


def discover_conditions(root):
    present = [d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "manifest_keypoints.csv"))]
    ordered = [c for c in CONDITION_ORDER if c in present]
    return ordered + sorted(c for c in present if c not in ordered)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = []
    for spec in args.config:
        name, paths = spec.split("=", 1)
        classifier_path, _, refiner_path = paths.partition(":")
        classifier, num_classes = load_classifier(classifier_path, device)
        refiner = load_refiner(refiner_path, device) if refiner_path else None
        configs.append((name, classifier, num_classes, refiner))

    conditions = discover_conditions(args.keypoint_root)
    print(f"Conditions: {', '.join(conditions)}")
    print(f"Configurations: {', '.join(c[0] for c in configs)}")

    results = []
    for condition in conditions:
        rows = read_keypoint_manifest(os.path.join(args.keypoint_root, condition, "manifest_keypoints.csv"), args.split)
        for name, classifier, num_classes, refiner in configs:
            arrays = refine_rows(rows, refiner, device) if refiner is not None else None
            dataset = KeypointSequenceDataset(rows, arrays=arrays)
            metrics = classification_metrics(*predict_logits(classifier, DataLoader(dataset, batch_size=64), device), num_classes)
            results.append({"condition": condition, "config": name, **metrics})
            print(f"{condition:20s} {name:28s} top1 {metrics['top1']:.3f} top5 {metrics['top5']:.3f} f1 {metrics['macro_f1']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    names = [c[0] for c in configs]
    lookup = {(r["condition"], r["config"]): r for r in results}
    lines = [
        f"Top-1 accuracy on the {args.split} split (top-5 in brackets)",
        "",
        "| Condition | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---" for _ in names) + " |",
    ]
    for condition in conditions:
        cells = [f"{lookup[(condition, n)]['top1'] * 100:.1f}% ({lookup[(condition, n)]['top5'] * 100:.1f}%)" for n in names]
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    md_path = os.path.splitext(args.out)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Results written to {args.out} and {md_path}")


if __name__ == "__main__":
    main()
