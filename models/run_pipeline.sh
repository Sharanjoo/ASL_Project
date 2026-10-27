#!/usr/bin/env bash
# Stages 1-3 + robustness table, end to end, from one Stage 1 keypoint checkpoint.
# Run from the asl_project folder:
#   bash models/run_pipeline.sh runs/keypoints_resnet50_256x192/best.pt runs/pipeline
# Args: <keypoint checkpoint> <output root> [frames manifest]
# Shorter CPU run: REFINER_EPOCHS=10 CLASSIFIER_EPOCHS=60 bash models/run_pipeline.sh ...
# Re-run only Stages 2-3 on keypoints already extracted into <output root>: SKIP_KEYPOINTS=1
set -euo pipefail

CKPT="$1"
ROOT="$2"
FRAMES="${3:-subset_100/frames16/manifest_frames.csv}"
KP="$ROOT/keypoints"
PY="${PYTHON:-python}"
REFINER_EPOCHS="${REFINER_EPOCHS:-40}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-150}"

# 1. Stage 1 keypoints on clean frames, then on every degradation x severity.
if [ -z "${SKIP_KEYPOINTS:-}" ]; then
    $PY models/keypoints/predict_wlasl.py --checkpoint "$CKPT" --manifest "$FRAMES" --out_dir "$KP/clean" --overlay_count 4
fi
NOISY=()
for d in occlusion blur missing; do
    for s in mild moderate severe; do
        if [ -z "${SKIP_KEYPOINTS:-}" ]; then
            $PY models/keypoints/predict_wlasl.py --checkpoint "$CKPT" --manifest "$FRAMES" \
                --out_dir "$KP/${d}_${s}" --degradation "$d" --severity "$s" \
                --hand_keypoints_dir "$KP/clean" --overlay_count 1
        fi
        NOISY+=("$KP/${d}_${s}/manifest_keypoints.csv")
    done
done

# 2. Stage 2 refiners: Baseline B (keypoint-space corruption only) and Proposed (visual degradations).
$PY models/refinement/train_refinement.py --mode synthetic --clean "$KP/clean/manifest_keypoints.csv" \
    --noisy "${NOISY[@]}" --out_dir "$ROOT/refiner_synthetic" --epochs "$REFINER_EPOCHS"
$PY models/refinement/train_refinement.py --mode degraded --clean "$KP/clean/manifest_keypoints.csv" \
    --noisy "${NOISY[@]}" --out_dir "$ROOT/refiner_degraded" --epochs "$REFINER_EPOCHS"

# 3. Stage 3 classifiers, each trained on clean-frame keypoints as its configuration sees them.
$PY models/recognition/train_recognition.py --keypoints "$KP/clean/manifest_keypoints.csv" --out_dir "$ROOT/classifier_frame_only" --epochs "$CLASSIFIER_EPOCHS"
for r in synthetic degraded; do
    $PY models/refinement/refine_keypoints.py --checkpoint "$ROOT/refiner_$r/best.pt" \
        --keypoints "$KP/clean/manifest_keypoints.csv" --out_dir "$ROOT/keypoints_refined_$r/clean"
    $PY models/recognition/train_recognition.py --keypoints "$ROOT/keypoints_refined_$r/clean/manifest_keypoints.csv" \
        --out_dir "$ROOT/classifier_refined_$r" --epochs "$CLASSIFIER_EPOCHS"
done

# 4. Robustness table on the test split.
$PY models/evaluate_robustness.py --keypoint_root "$KP" --out "$ROOT/robustness.csv" \
    --config "Frame-only=$ROOT/classifier_frame_only/best.pt" \
    --config "+Temporal=$ROOT/classifier_refined_synthetic/best.pt:$ROOT/refiner_synthetic/best.pt" \
    --config "+Temporal & robust training=$ROOT/classifier_refined_degraded/best.pt:$ROOT/refiner_degraded/best.pt"

cat "$ROOT/robustness.md"
