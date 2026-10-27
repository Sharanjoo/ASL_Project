# Running the whole ASL pipeline on the SJSU GPU machine

One script, `run_all_gpu.sh`, runs every stage in order:

1. Restore the WLASL clips
2. Download COCO 2017 + COCO-WholeBody annotations (~20 GB)
3. Train the Stage 1 keypoint extractor (ResNet-50, 256×192, 140 epochs) — the long step
4. Sample 16 frames per WLASL clip
5. Extract keypoints under all 10 conditions (clean + 3 degradations × 3 severities),
   train both Stage 2 refiners and all three Stage 3 classifiers, and write the
   robustness table

It is safe to re-run at any point: finished steps are skipped and Stage 1 resumes
from its last checkpoint.

## What you need

| File | Size | What it is |
|------|------|------------|
| `asl_project_code.zip` | < 100 KB | All code, this file included |
| `asl_wlasl_videos.zip` | ~540 MB | The 1,020 WLASL100 clips we could download + `subset_100/manifest.csv` |

On the GPU machine: Linux, Python 3.10+, `curl`, `unzip`, at least **25 GB free disk**,
and a GPU with ≥ 8 GB memory.

The WLASL clips fall under the dataset's C-UDA license (in `WLASL/start_kit/C-UDA-1.0.pdf`
of the WLASL repo) — keep them within the team and don't post them publicly.

## Step 1 — Unzip both files into one folder

```bash
unzip asl_project_code.zip
unzip asl_wlasl_videos.zip
cd asl_project
ls
```

You should see `run_all_gpu.sh`, `GPU_RUN.md`, `models/`, `data/`, `raw_videos/` and
`subset_100/`. Both zips unpack into the same `asl_project/` folder.

## Step 2 — Install Python packages with GPU support

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the last line prints `False`, the CPU build of PyTorch was installed. Check the
CUDA version with `nvidia-smi` (top-right corner) and reinstall the matching build from
https://pytorch.org/get-started/locally/, for example for CUDA 12.1:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If the cluster uses modules, you may need `module load cuda` (or similar) first —
check the SJSU HPC documentation.

## Step 3 — Quick check (about 5–10 minutes, after the data download)

Before committing a day of GPU time, confirm Stage 1 trains and measure its speed.
The first line downloads the COCO data (~20 GB, the slow part); the second runs
50 training steps:

```bash
bash models/keypoints/setup_gpu_data.sh
bash models/keypoints/train_gpu.sh --epochs 1 --max_steps_per_epoch 50 --max_eval_batches 20 --out_dir runs/gpu_check
```

It prints the seconds the epoch took. A full epoch is roughly
`(training people ÷ 64) ÷ 50 ×` that time; multiply by 140 for the full run. If it
fails with `CUDA out of memory`, see Troubleshooting.

## Step 4 — Full run

Directly on the machine (use `tmux` or `screen` so it survives disconnecting):

```bash
tmux new -s asl
source .venv/bin/activate
bash run_all_gpu.sh 2>&1 | tee run_all_gpu.log
```

Detach with `Ctrl-b d`, reattach later with `tmux attach -t asl`.

On a Slurm cluster, submit it as a job instead:

```bash
sbatch --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=48:00:00 \
    --output=run_all_gpu_%j.log --wrap "source .venv/bin/activate && bash run_all_gpu.sh"
```

If the job hits its time limit, submit the same command again — it picks up where it
stopped.

Expected time: Stage 1 training dominates (likely 10–30 hours depending on the GPU;
step 3's estimate is the real number). Everything after it takes about 1 hour on a GPU.

## Step 5 — What to send back

Zip these and share them with the team (about 0.5 GB, mostly the Stage 1 `best.pt`,
which also stores optimizer state for resuming):

```bash
zip -r asl_gpu_results.zip \
    runs/keypoints_resnet50_256x192/best.pt \
    runs/keypoints_resnet50_256x192/train_log.csv \
    runs/keypoints_resnet50_256x192/final_metrics.json \
    runs/pipeline/robustness.md runs/pipeline/robustness.csv \
    runs/pipeline/refiner_*/final_metrics.json runs/pipeline/refiner_*/best.pt \
    runs/pipeline/classifier_*/final_metrics.json runs/pipeline/classifier_*/best.pt \
    runs/pipeline/keypoints/clean/*.png \
    run_all_gpu.log
```

| File | What it tells us |
|------|------------------|
| `keypoints_resnet50_256x192/final_metrics.json` | Stage 1 PCK and error per body part (upper body, mouth, hands) |
| `pipeline/keypoints/clean/overlay_*.png` | Keypoints drawn on WLASL frames — the domain-gap eyeball check |
| `pipeline/refiner_*/final_metrics.json` | Stage 2 keypoint error vs. no refinement |
| `pipeline/classifier_*/final_metrics.json` | Stage 3 top-1 / top-5 / F1 on clean test clips |
| `pipeline/robustness.md` | The proposal's robustness table, filled in |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `torch.cuda.is_available()` is `False` | Reinstall the CUDA build of PyTorch (step 2) |
| `CUDA out of memory` in Stage 1 | Edit `models/keypoints/train_gpu.sh`: `--batch_size 32` (then 16) |
| Data loading is slow / GPU mostly idle | Raise `--num_workers` in `train_gpu.sh` to the CPU cores you have |
| `gdown` fails on the annotation download | Download `coco_wholebody_train_v1.0.json` / `coco_wholebody_val_v1.0.json` by hand from https://github.com/jin-s13/COCO-WholeBody into `coco_wholebody_ann/`, then re-run |
| `Missing subset_100/manifest.csv or raw_videos/` | `asl_wlasl_videos.zip` wasn't unzipped into the `asl_project/` folder |
| Want to stop Stage 1 early | Stop it anytime; `best.pt` is always the best epoch so far. Then run `SKIP_STAGE1_TRAINING=1 bash run_all_gpu.sh` to do the remaining steps with it |

Questions: see the full pipeline description in `README.md`.
