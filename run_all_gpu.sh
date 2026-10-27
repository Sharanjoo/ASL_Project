#!/usr/bin/env bash
# Every stage on the GPU machine, in order. See GPU_RUN.md.
# Run from the asl_project folder:  bash run_all_gpu.sh
# Safe to re-run after a crash or job time limit: finished steps are skipped and
# Stage 1 training resumes from its last checkpoint.
# Extra Stage 1 flags: STAGE1_ARGS="--num_workers 24" bash run_all_gpu.sh
set -euo pipefail

STAGE1_DIR=runs/keypoints_resnet50_256x192
PIPELINE_DIR=runs/pipeline

log() { echo; echo "=== [$(date '+%H:%M:%S')] $* ==="; }

log "Checking GPU"
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see a GPU - install the CUDA build of torch'; print('GPU:', torch.cuda.get_device_name(0))"

log "Step 1/5: WLASL videos"
if [ ! -f subset_100/manifest.csv ] || [ ! -d raw_videos/youtube_full ]; then
    echo "Missing subset_100/manifest.csv or raw_videos/. Unzip asl_wlasl_videos.zip into this folder first (GPU_RUN.md, step 2)."
    exit 1
fi
python data/wlasl/restore_youtube_links.py --manifest subset_100/manifest.csv --video_dir raw_videos

log "Step 2/5: COCO-WholeBody data (~20 GB, skipped if present)"
bash models/keypoints/setup_gpu_data.sh

log "Step 3/5: Stage 1 keypoint extractor training"
if [ -n "${SKIP_STAGE1_TRAINING:-}" ]; then
    echo "SKIP_STAGE1_TRAINING set: using $STAGE1_DIR/best.pt as it stands"
    [ -f "$STAGE1_DIR/best.pt" ] || { echo "No $STAGE1_DIR/best.pt yet"; exit 1; }
elif [ -f "$STAGE1_DIR/final_metrics.json" ]; then
    echo "Already trained: $STAGE1_DIR/final_metrics.json exists"
elif [ -f "$STAGE1_DIR/last.pt" ]; then
    bash models/keypoints/train_gpu.sh ${STAGE1_ARGS:-} --resume "$STAGE1_DIR/last.pt"
else
    bash models/keypoints/train_gpu.sh ${STAGE1_ARGS:-}
fi

log "Step 4/5: Sample 16 frames per WLASL clip"
if [ -f subset_100/frames16/manifest_frames.csv ]; then
    echo "Already sampled: subset_100/frames16/manifest_frames.csv exists"
else
    python data/wlasl/sample_frames.py --manifest subset_100/manifest.csv --video_dir raw_videos \
        --out_dir subset_100/frames16 --num_frames 16 --img_size 224
fi

log "Step 5/5: Degraded keypoints, Stage 2 refiners, Stage 3 classifiers, robustness table"
bash models/run_pipeline.sh "$STAGE1_DIR/best.pt" "$PIPELINE_DIR"

log "Done. Send back the files listed in GPU_RUN.md, step 5"
