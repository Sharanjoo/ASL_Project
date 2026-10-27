#!/usr/bin/env bash
# Full Stage 1 training run: train on COCO-WholeBody train2017, evaluate on val2017.
# Run from the asl_project folder:  bash models/keypoints/train_gpu.sh
# On a Slurm cluster, submit it instead, e.g.:
#   sbatch --gres=gpu:1 --cpus-per-task=8 --time=24:00:00 models/keypoints/train_gpu.sh
set -euo pipefail

python -u models/keypoints/train_keypoints.py \
    --train_ann coco_wholebody_ann/coco_wholebody_train_v1.0.json \
    --train_img_dir coco_images/train2017 \
    --val_ann coco_wholebody_ann/coco_wholebody_val_v1.0.json \
    --val_img_dir coco_images/val2017 \
    --out_dir runs/keypoints_resnet50_256x192 \
    --backbone resnet50 \
    --input_h 256 \
    --input_w 192 \
    --epochs 140 \
    --batch_size 64 \
    --num_workers 8 \
    "$@"
