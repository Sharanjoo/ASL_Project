# Comparing our Stage 1 with MediaPipe and RTMW

Group 2, DATA 255. Numbers come from `report/results/` (mean ± std over 5 seeds unless stated). Raw per-seed tables are in `report/results/baselines/`.

## What was compared

Our Stage 1 keypoint extractor against two off-the-shelf extractors, with everything else held fixed:

- **MediaPipe** (Holistic landmarker): pose, face mesh and two hands, run per frame in image mode.
- **RTMW** (whole-body RTMPose family, `rtmw-dw-x-l`, run through `rtmlib`): used as the stand-in for OpenPose, which is hard to install on current CUDA. Its published weights were trained on a larger and more varied mix of datasets than our Stage 1, which uses COCO-WholeBody only (the weights are labelled "cocktail14"). It is not a like-for-like architecture comparison.

## Setup

- **Same frames, same damage.** Every extractor sees the same 16-frame arrays, degraded by the same seeded code (occlusion, motion blur, missing frames at three severities). Occluders sit on the Stage 1 clean wrist positions for all models.
- **Same layout.** Each extractor's output is mapped onto our 73 points (11 upper body, 20 mouth, 21 per hand). Undetected points get score 0 and position (0, 0). MediaPipe's face-mesh vertices were hand-mapped to the 20 iBUG mouth points, and its face and hand points have no real confidence, so detected points get score 1.0. RTMW already uses the COCO-WholeBody ordering.
- **Same Stage 3 recipe.** A bi-GRU classifier is trained on each extractor's clean-condition keypoints (150 epochs, 5 seeds) and evaluated on all 10 conditions. The baselines are not tuned.
- **Two settings of Stage 3:** plain, and "+Interpolation" (low-confidence points replaced by interpolation, no training).
- Test set: the 258 official WLASL100 test clips. One clip is about 0.4 points.

## 1. Keypoint accuracy

**100 hand-labeled WLASL frames** (90 scored; error as a fraction of shoulder width; PCK = share of points within the threshold):

| | PCK@0.1 | PCK@0.2 | Mean error | Left / right hand PCK@0.2 |
|---|---|---|---|---|
| RTMW | 77.3% | **99.0%** | 0.063 | 99.6 / 99.4 |
| Stage 1 (ours) | 71.7% | 97.6% | 0.074 | 96.5 / 97.2 |
| MediaPipe | 66.1% | 95.4% | 0.127 | 90.7 / 92.0 |

**COCO-WholeBody val** (3,545 person instances, Stage 1's own protocol: GT-box crop, error divided by box side, undetected counts as a miss):

| | PCK@0.05 | PCK@0.1 | NME |
|---|---|---|---|
| RTMW | **92.2%** | **96.5%** | **0.025** |
| Stage 1 (ours) | 90.4% | 95.2% | 0.030 |
| MediaPipe | 41.8% | 46.2% | 0.404 |

RTMW is the most accurate on both checks, and Stage 1 is close behind. MediaPipe is far behind on COCO val (hands 35.8% / 35.5% at PCK@0.1). Part of that gap comes from the protocol: COCO person crops have small hands that MediaPipe often fails to detect, and a missed point counts as a miss. On the signer-centred WLASL frames MediaPipe does much better (PCK@0.2 95.4%), but its hands are still the weakest.

## 2. Sign recognition (top-1 %, mean ± std over 5 seeds)

| Condition | Stage 1 Frame-only | Stage 1 Proposed (+refiner) | Raw-frame CNN | MediaPipe | MediaPipe +Interp. | RTMW | RTMW +Interp. |
|---|---|---|---|---|---|---|---|
| Clean | 49.5 ± 1.3 | 48.8 ± 2.6 | 39.0 ± 1.6 | 37.0 ± 3.1 | 27.1 ± 2.8 | **53.8 ± 2.1** | **53.8 ± 2.0** |
| Occlusion, severe | 31.7 ± 0.8 | 36.4 ± 1.3 | 23.4 ± 3.1 | 19.7 ± 1.6 | 19.9 ± 1.5 | **37.9 ± 2.3** | 37.4 ± 2.2 |
| Blur, severe | 31.5 ± 2.1 | **40.0 ± 1.2** | 16.9 ± 3.5 | 8.4 ± 0.5 | 19.8 ± 1.6 | 37.4 ± 2.8 | 37.2 ± 2.6 |
| Missing, moderate | 31.7 ± 3.3 | **42.8 ± 1.9** | 32.6 ± 1.1 | 24.8 ± 2.2 | 25.8 ± 1.4 | 41.4 ± 3.2 | 31.2 ± 4.5 |
| Missing, severe | 13.8 ± 3.1 | **29.5 ± 2.2** | 20.5 ± 0.9 | 13.7 ± 1.3 | 17.9 ± 2.2 | 23.7 ± 3.8 | 14.8 ± 3.9 |
| **Average, 9 degraded** | 36.6 ± 1.3 | 42.2 ± 1.7 | 29.9 ± 1.2 | 23.6 ± 0.7 | 23.6 ± 1.7 | **43.9 ± 1.8** | 41.1 ± 1.8 |
| Drop, clean to degraded | −12.9 | **−6.6** | −9.1 | −13.4 | −3.5 | −9.9 | −12.7 |

All 10 conditions, plus top-5 and macro-F1: `report/results/baselines/robustness_mean_std_all_*.md`.

## 3. What the numbers say

1. **RTMW is the strongest extractor.** With the same plain Stage 3, RTMW beats our Stage 1 on clean clips (53.8 vs 49.5) and on the degraded average (43.9 vs 36.6). The clean gap is about two standard deviations, and the paired bootstrap in section 4 agrees (+4.3, interval [+0.4, +8.4]).
2. **Our Stage 1 plus the robust refiner roughly matches RTMW under damage.** Proposed averages 42.2 against RTMW's 43.9, a gap smaller than one standard deviation, so we cannot separate them. Proposed is ahead where damage is heaviest on frames (missing moderate 42.8 vs 41.4, missing severe 29.5 vs 23.7, severe blur 40.0 vs 37.4), and RTMW is ahead on clean clips and occlusion (the severe-occlusion gap is within noise). Note that this row compares a pipeline with a refiner to an extractor without one. The like-for-like extractor comparison is the Frame-only column against MediaPipe and RTMW.
3. **The refiner does real work.** It adds about 5.6 points on average to our Stage 1 and halves its drop under degradation. We did not apply the refiner to RTMW keypoints, so we do not know whether it would help a stronger extractor too. That is the obvious next experiment.
4. **MediaPipe is the weakest extractor here.** Its average (23.6) is below the raw-frame CNN (29.9), and it collapses under severe blur (8.4). Interpolation helps it under blur (19.8 at severe) but costs it on clean clips (37.0 to 27.1).
5. **Interpolation does not help RTMW** and hurts under missing frames (missing moderate 41.4 to 31.2). We did not investigate why.
6. **Keypoint accuracy and sign accuracy agree in order** (RTMW > Stage 1 > MediaPipe), but not in proportion. RTMW's 1.3-point PCK@0.1 lead over Stage 1 on COCO val comes with a 4.3-point top-1 lead, while MediaPipe's 49-point deficit on COCO val costs only 12.5 points of top-1 on clean clips. COCO val tests small, hard-to-see hands, which is a harder setting than the signer-centred WLASL frames.

## 4. Follow-up checks (done after the first comparison)

**Confidence intervals** (`report/results/baselines/bootstrap_ci.md`; 10,000 resamples of the 258 test clips, paired between models, each clip scored as the share of 5 seeds that get it right):

- The refiner's gain over Stage 1 Frame-only on the degraded average is +5.6 points, 95% interval [+3.9, +7.4]. It is +15.7 [+11.6, +19.8] on severe missing frames, and −0.6 [−2.8, +1.6] on clean clips (no gain, no loss).
- RTMW over Frame-only on clean clips is +4.3 [+0.4, +8.4], real but borderline.
- RTMW minus Proposed on the degraded average is +1.7 [−1.0, +4.4]: not distinguishable. On severe missing frames Proposed leads RTMW by 5.7 [1.6, 9.8]. On clean clips RTMW leads Proposed by 5.0 [+1.2, +8.7]. Severe blur and severe occlusion are not distinguishable.
- These intervals cover clip-to-clip variation only. They do not cover Stage 1 training variation (see caveats).

**MediaPipe confidence scores** (5 seeds, untuned Stage 3, `robustness_mean_std_with_variants_tuned_*`): the first comparison gave MediaPipe's hand and mouth points a score of 1.0. Two alternatives were tested, scores taken from pose-wrist/nose visibility, and all detected points set to a flat 1.0.

| | Clean | Average, 9 degraded |
|---|---|---|
| MediaPipe (as in the first comparison) | 37.0 ± 3.1 | 23.6 ± 0.7 |
| MediaPipe, visibility scores | 38.8 ± 2.1 | 24.2 ± 1.1 |
| MediaPipe, flat scores | 37.8 ± 1.7 | 23.7 ± 0.5 |

All three are within one standard deviation of each other, so the placeholder scores are not why MediaPipe trails. Hand detection is.

**Stage 3 tuning, chosen on validation only.** Grid of hidden size {128, 256}, dropout {0.3, 0.5}, learning rate {1e-3, 3e-4}, two seeds per setting, best mean validation top-1 retrained with 5 seeds (`report/results/baselines/tuning/`). The selected setting was learning rate 3e-4 for all three extractors (hidden 256 for Stage 1 and RTMW, 128 for MediaPipe; dropout 0.3).

| Top-1 % (mean ± std, 5 seeds) | Clean, untuned | Clean, tuned | Degraded avg, untuned | Degraded avg, tuned |
|---|---|---|---|---|
| Stage 1 Frame-only | 49.5 ± 1.3 | 59.6 ± 1.7 | 36.6 ± 1.3 | 44.1 ± 1.9 |
| RTMW | 53.8 ± 2.1 | 62.2 ± 1.3 | 43.9 ± 1.8 | 49.2 ± 0.6 |
| MediaPipe | 37.0 ± 3.1 | 51.6 ± 1.3 | 23.6 ± 0.7 | 32.0 ± 1.7 |

What this changes:

- **Tuning helps every extractor a lot, mostly through the lower learning rate**, and the order does not change: RTMW > Stage 1 > MediaPipe, on clean and degraded alike.
- **The gaps shrank but remain.** RTMW's lead over Stage 1 on the degraded average is 5.1 points tuned (49.2 vs 44.1) against 7.3 untuned. On clean clips it is 2.6 tuned against 4.3 untuned. MediaPipe's gap to Stage 1 on clean clips shrank from 12.5 to 8.0. We did not recompute confidence intervals for the tuned models.
- **The headline "Proposed roughly matches RTMW under damage" does not survive tuning as stated.** The tuned Frame-only classifier (44.1) already matches the untuned Proposed pipeline (42.2). The refiner pipeline was not tuned, so the fair comparison is tuned Frame-only against tuned RTMW (44.1 vs 49.2), and we have not run tuned Stage 3 on the refined keypoints. Until we do, the claim to make is that the refiner helps Stage 1 under the same untuned recipe, not that it matches a tuned RTMW.
- **Tuned models lose more on severe missing frames.** Tuned Frame-only and RTMW score 14.7 and 18.1 there, below untuned Proposed (29.5). Tuning improved clean-clip fitting, not robustness to missing frames.
- The best setting sits at the edge of the grid (lowest learning rate), so a wider search might improve all three further. The validation split has 338 clips, so differences of a point or two between settings are noise.

## 5. Caveats

- **MediaPipe is a fair but imperfect baseline.** The mouth mapping used hand-chosen face-mesh points, its hand and face points have no real confidence score, and it ran per frame in image mode with no tracking. Its mouth accuracy on the labeled frames is at least on par with the others (PCK@0.2 98.9% against 97.7% for both Stage 1 and RTMW), so the mapping is probably not the main problem; hand detection is.
- **RTMW stands in for OpenPose.** We did not run OpenPose itself.
- **One trained Stage 1.** Seeds vary Stages 2-3 and the classifiers, not Stage 1. MediaPipe and RTMW are fixed models.
- **Tuning only partly applied.** The main tables use one untuned Stage 3 recipe for every extractor. Tuned results (section 4) exist for Frame-only, RTMW and MediaPipe, not for the refiner pipeline, so Proposed is not compared against tuned RTMW.
- **Small test set.** 258 clips. Differences within about one standard deviation are not claimed. Clip-level intervals are in section 4 for the untuned models.
- Results are on one dataset (WLASL100) whose official splits are not signer-disjoint.

## 6. Still to do

Done: paired bootstrap intervals, MediaPipe score variants, per-extractor Stage 3 tuning. Remaining: two extra Stage 1 training seeds (to measure Stage 1's own run-to-run spread); tuned Stage 3 on the refined keypoints so the refiner pipeline can be compared with tuned RTMW; a wider learning-rate search. This document will be updated when they are done.

## Reproduce

```bash
# per-extractor keypoints for all 10 conditions (own venv per extractor), then Stage 3 per seed
python models/baselines/extract_baseline.py --model mediapipe --workers 12
python models/baselines/extract_baseline.py --model rtmw --device cuda --workers 4
bash models/baselines/run_stage3.sh rtmw "RTMW" "0 1 2 3 4"
# keypoint accuracy
python models/baselines/predict_label_set.py --model rtmw --out runs/baselines/rtmw/label_preds.npy
python models/baselines/eval_coco.py --model rtmw --device cuda --workers 2
```
