# Results tables behind the progress-checkup deck

Small copies of the final numbers, so the deck can be rebuilt from the repo. The full outputs (checkpoints, keypoints, logs) stay in the git-ignored `runs/` folder.

All numbers are top-1 / top-5 / macro-F1 on the 258 WLASL100 test clips, 10 conditions (clean plus occlusion, blur and missing frames at mild, moderate and severe), 5 seeds (0-4).

| File | What it is |
|---|---|
| `robustness_mean_std_top1.md` / `.csv` | Mean ± std over the 5 seeds, top-1. Also `_top5` and `_macro_f1`. The deck reads the top-1 CSV. |
| `seed0_robustness.csv` / `.md` | Seed 0: Frame-only, +Interpolation, +Temporal, Proposed and the raw-frame CNN (from `runs/pipeline`). |
| `seeds/seed1..4/` | Seeds 1-4: `robustness.csv` (keypoint models) and `robustness_raw_frame.csv`. |
| `stage1_coco_val_metrics.json` | Stage 1 on COCO-WholeBody val (PCK, NME by group). |
| `stage1_domain_gap_metrics.json` | Stage 1 on the 100 hand-labeled WLASL frames. |
| `confusion_matrix_*.png` | Clean-condition confusion matrices on the test split. |

`baselines/` holds the MediaPipe and RTMW results (see `../baseline_comparison.md`): per-seed tables in `baselines/seeds/`, the combined mean ± std tables in `baselines/robustness_mean_std_all_*`, and keypoint-accuracy metrics (`*_domain_gap_metrics.json` on the 100 hand-labeled WLASL frames, `*_coco_val_metrics.json` on COCO-WholeBody val).

Regenerate the mean ± std tables (run from the repo root):

```bash
python models/aggregate_seeds.py --main report/results/seed0_robustness.csv --seeds_dir report/results/seeds --out report/results/robustness_mean_std --metric top1
```

Rebuild the deck (needs Node with `pptxgenjs` and `jszip` on `NODE_PATH`):

```bash
node report/build_midsem_deck.js
```
