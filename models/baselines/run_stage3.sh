#!/usr/bin/env bash
# Stage 3 on a baseline's keypoints, same recipe as the main pipeline, one classifier per seed.
#   PYTHON=.venv/Scripts/python.exe bash models/baselines/run_stage3.sh mediapipe "MediaPipe" "0 1 2 3 4"
# Writes runs/seeds/seed<N>/robustness_<model>.csv (read by models/aggregate_seeds.py).
# No `set -e`: on Windows the training script crashes at interpreter exit after saving its outputs.
set -uo pipefail

MODEL="$1"
NAME="$2"
SEEDS="${3:-0 1 2 3 4}"
PY="${PYTHON:-python}"
KP="runs/baselines/$MODEL/keypoints"
EPOCHS="${CLASSIFIER_EPOCHS:-150}"

for SEED in $SEEDS; do
    OUT="runs/baselines/$MODEL/classifier_seed$SEED"
    mkdir -p "runs/seeds/seed$SEED"
    echo "=== $MODEL stage3 seed $SEED ($(date +%H:%M:%S))"
    $PY models/recognition/train_recognition.py --keypoints "$KP/clean/manifest_keypoints.csv" \
        --out_dir "$OUT" --epochs "$EPOCHS" --seed "$SEED"
    $PY models/evaluate_robustness.py --keypoint_root "$KP" --out "runs/seeds/seed$SEED/robustness_$MODEL.csv" \
        --config "$NAME=$OUT/best.pt" \
        --config "$NAME +Interpolation=$OUT/best.pt:interpolate"
    echo "=== done $MODEL stage3 seed $SEED ($(date +%H:%M:%S))"
done
