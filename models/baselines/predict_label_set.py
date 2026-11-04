"""Predict the 73-point layout on the 100 hand-labeled WLASL frames, ready for score_domain_gap.py --predictions.

  .venv_mp/Scripts/python models/baselines/predict_label_set.py --model mediapipe --out runs/baselines/mediapipe/label_set_preds.npy
  .venv/Scripts/python models/keypoints/domain_gap/score_domain_gap.py --predictions <that .npy> \
      --label_set domain_gap/label_set.npz --labels domain_gap/domain_gap_labels_Sharan.json
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ESTIMATORS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(ESTIMATORS))
    p.add_argument("--label_set", default="domain_gap/label_set.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--swap_hands", action="store_true", help="mediapipe only: test the left/right hand convention")
    args = p.parse_args()

    frames = np.load(args.label_set)["frames"]
    kwargs = {"device": args.device} if args.model == "rtmw" else {"swap_hands": args.swap_hands}
    estimator = ESTIMATORS[args.model](**kwargs)
    preds = np.concatenate([estimator(frames[i:i + 1]) for i in range(len(frames))])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, preds)
    detected = (preds[..., 2] > 0)
    print(f"{args.model}: {len(preds)} frames -> {args.out}; points detected: "
          f"body {detected[:, :11].mean():.0%}, mouth {detected[:, 11:31].mean():.0%}, "
          f"left hand {detected[:, 31:52].mean():.0%}, right hand {detected[:, 52:].mean():.0%}")


if __name__ == "__main__":
    main()
