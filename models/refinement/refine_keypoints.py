import argparse
import csv
import os

import numpy as np
import torch

from refiner import TemporalRefiner
from windows import make_windows, to_model_space, to_pixel_space


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a trained Stage 2 refiner to a keypoint directory, writing a refined copy."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--keypoints", required=True, help="manifest_keypoints.csv to refine")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    model = TemporalRefiner(arch=train_args["arch"], hidden=train_args["hidden"], num_layers=train_args["num_layers"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()

    with open(args.keypoints, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    out_rows = []
    with torch.no_grad():
        for row in rows:
            sequence = to_model_space(np.load(row["keypoints_path"]))
            refined = model(torch.from_numpy(make_windows(sequence)).to(device)).cpu().numpy()
            out_path = os.path.join(args.out_dir, f"{row['video_id']}.npy")
            np.save(out_path, to_pixel_space(refined))
            out_rows.append({**row, "keypoints_path": out_path})

    manifest_path = os.path.join(args.out_dir, "manifest_keypoints.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Clips refined: {len(out_rows)}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
