import argparse
import csv
import os
import sys

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "recognition"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "data", "degradation"))

from degrade import DEGRADATIONS, SEVERITIES  # noqa: E402
from frame_dataset import FrameClipDataset, read_frame_manifest  # noqa: E402
from frame_model import FrameGRUClassifier  # noqa: E402
from metrics import classification_metrics, predict_logits  # noqa: E402

CONDITIONS = [("clean", "moderate")] + [(d, s) for d in DEGRADATIONS[1:] for s in SEVERITIES]
CONDITION_ORDER = ["clean"] + [f"{d}_{s}" for d in DEGRADATIONS[1:] for s in SEVERITIES]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score the raw-frame baseline on all 10 conditions, degrading the frames on the fly "
                    "with the same seeded damage the keypoint pipeline sees."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="manifest_frames.csv from sample_frames.py")
    parser.add_argument("--hand_keypoints_dir", help="Clean Stage 1 keypoint dir (runs/pipeline/keypoints/clean): puts the occluder on the same wrists")
    parser.add_argument("--name", default="Raw-frame CNN")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True, help="Output CSV; a .md table is written next to it")
    parser.add_argument("--merge_into", help="Existing robustness.csv: append this column to it and rewrite its .md")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def write_table(results, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "config", "top1", "top5", "macro_f1", "n"])
        writer.writeheader()
        writer.writerows(results)

    names = list(dict.fromkeys(r["config"] for r in results))
    present = {r["condition"] for r in results}
    conditions = [c for c in CONDITION_ORDER if c in present] + sorted(present - set(CONDITION_ORDER))
    lookup = {(r["condition"], r["config"]): r for r in results}
    lines = [
        "Top-1 accuracy on the test split (top-5 in brackets)",
        "",
        "| Condition | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---" for _ in names) + " |",
    ]
    for condition in conditions:
        cells = []
        for n in names:
            r = lookup.get((condition, n))
            cells.append(f"{float(r['top1']) * 100:.1f}% ({float(r['top5']) * 100:.1f}%)" if r else "")
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    with open(os.path.splitext(out_csv)[0] + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    a = checkpoint["args"]
    model = FrameGRUClassifier(a["num_classes"], a["hidden"], a["num_layers"], a["dropout"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()

    rows = read_frame_manifest(args.manifest, args.split)
    print(f"{args.split} clips: {len(rows)}")

    results = []
    for degradation, severity in CONDITIONS:
        condition = degradation if degradation == "clean" else f"{degradation}_{severity}"
        dataset = FrameClipDataset(rows, degradation=degradation, severity=severity, hand_keypoints_dir=args.hand_keypoints_dir)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        metrics = classification_metrics(*predict_logits(model, loader, device), a["num_classes"])
        results.append({"condition": condition, "config": args.name, **metrics})
        print(f"{condition:20s} {args.name:20s} top1 {metrics['top1']:.3f} top5 {metrics['top5']:.3f} f1 {metrics['macro_f1']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_table(results, args.out)
    print(f"Results written to {args.out}")

    if args.merge_into:
        with open(args.merge_into, "r", newline="", encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r["config"] != args.name]
        write_table(existing + results, args.merge_into)
        print(f"Column '{args.name}' merged into {args.merge_into} (and its .md)")


if __name__ == "__main__":
    main()
