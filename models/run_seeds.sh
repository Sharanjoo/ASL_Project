#!/usr/bin/env bash
# Extra seeds for Stages 2-3 (keypoint) or the raw-frame CNN. Seed 0 is the main run in runs/pipeline.
# Stage 1 keypoints in runs/pipeline/keypoints are reused as they are (one Stage 1 checkpoint, fixed degradations).
#   bash models/run_seeds.sh keypoint "1 2 3 4"     # refiners + classifiers + robustness table per seed
#   bash models/run_seeds.sh raw "1 2 3 4"          # raw-frame CNN train + eval per seed
# No `set -e`: on Windows the training scripts crash at interpreter exit (0xC0000409) after saving their outputs.
set -uo pipefail

MODE="$1"
SEEDS="$2"
PY="${PYTHON:-python}"
KP="${KP:-runs/pipeline/keypoints}"
OUT="${OUT:-runs/seeds}"
FRAMES="${FRAMES:-subset_100/frames16/manifest_frames.csv}"
REFINER_EPOCHS="${REFINER_EPOCHS:-40}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-150}"
RAW_FRAME_EPOCHS="${RAW_FRAME_EPOCHS:-30}"

NOISY=()
for d in occlusion blur missing; do
    for s in mild moderate severe; do
        NOISY+=("$KP/${d}_${s}/manifest_keypoints.csv")
    done
done

for SEED in $SEEDS; do
    ROOT="$OUT/seed$SEED"
    mkdir -p "$ROOT"
    echo "=== $MODE seed $SEED -> $ROOT ($(date +%H:%M:%S))"

    if [ "$MODE" = "keypoint" ]; then
        for r in synthetic degraded; do
            $PY models/refinement/train_refinement.py --mode "$r" --clean "$KP/clean/manifest_keypoints.csv" \
                --noisy "${NOISY[@]}" --out_dir "$ROOT/refiner_$r" --epochs "$REFINER_EPOCHS" --seed "$SEED"
        done
        $PY models/recognition/train_recognition.py --keypoints "$KP/clean/manifest_keypoints.csv" \
            --out_dir "$ROOT/classifier_frame_only" --epochs "$CLASSIFIER_EPOCHS" --seed "$SEED"
        for r in synthetic degraded; do
            $PY models/refinement/refine_keypoints.py --checkpoint "$ROOT/refiner_$r/best.pt" \
                --keypoints "$KP/clean/manifest_keypoints.csv" --out_dir "$ROOT/keypoints_refined_$r/clean"
            $PY models/recognition/train_recognition.py --keypoints "$ROOT/keypoints_refined_$r/clean/manifest_keypoints.csv" \
                --out_dir "$ROOT/classifier_refined_$r" --epochs "$CLASSIFIER_EPOCHS" --seed "$SEED"
        done
        $PY models/evaluate_robustness.py --keypoint_root "$KP" --out "$ROOT/robustness.csv" \
            --config "Frame-only=$ROOT/classifier_frame_only/best.pt" \
            --config "+Interpolation=$ROOT/classifier_frame_only/best.pt:interpolate" \
            --config "+Temporal=$ROOT/classifier_refined_synthetic/best.pt:$ROOT/refiner_synthetic/best.pt" \
            --config "+Temporal & robust training=$ROOT/classifier_refined_degraded/best.pt:$ROOT/refiner_degraded/best.pt"
    else
        $PY models/raw_frame/train_raw_frames.py --manifest "$FRAMES" --out_dir "$ROOT/classifier_raw_frame" \
            --epochs "$RAW_FRAME_EPOCHS" --batch_size 32 --seed "$SEED"
        $PY models/raw_frame/evaluate_raw_frames.py --checkpoint "$ROOT/classifier_raw_frame/best.pt" --manifest "$FRAMES" \
            --hand_keypoints_dir "$KP/clean" --out "$ROOT/robustness_raw_frame.csv"
    fi
    echo "=== done $MODE seed $SEED ($(date +%H:%M:%S))"
done
