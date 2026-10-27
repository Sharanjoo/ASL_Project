#!/usr/bin/env bash
# Download COCO 2017 images + COCO-WholeBody annotations on the GPU machine.
# Run from the asl_project folder:  bash models/keypoints/setup_gpu_data.sh
# Needs ~20 GB free. Already-present files are skipped, so it is safe to re-run.
set -euo pipefail

pip install -q -r requirements.txt gdown

mkdir -p coco_images coco_wholebody_ann

for split in val2017 train2017; do
    if [ -d "coco_images/$split" ]; then
        echo "coco_images/$split already exists, skipping"
        continue
    fi
    curl -L -C - -o "coco_images/$split.zip" "http://images.cocodataset.org/zips/$split.zip"
    unzip -q "coco_images/$split.zip" -d coco_images
    rm "coco_images/$split.zip"
done

# Google Drive mirrors listed in https://github.com/jin-s13/COCO-WholeBody
if [ ! -f coco_wholebody_ann/coco_wholebody_val_v1.0.json ]; then
    python -m gdown 1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb -O coco_wholebody_ann/coco_wholebody_val_v1.0.json
fi
if [ ! -f coco_wholebody_ann/coco_wholebody_train_v1.0.json ]; then
    python -m gdown 1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf -O coco_wholebody_ann/coco_wholebody_train_v1.0.json
fi

echo "train2017 images: $(ls coco_images/train2017 | wc -l)"
echo "val2017 images:   $(ls coco_images/val2017 | wc -l)"
ls -lh coco_wholebody_ann
