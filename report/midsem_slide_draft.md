# Midsem presentation: slide text draft (Group 2, DATA 255)

> **Superseded by `report/midsem_deck.pptx`.** Raw-frame seed 4 finished after this draft was written, so the raw-frame
> numbers below (marked ◆) are out of date. The deck is built from the final 5-seed file by `report/build_midsem_deck.js`.

Source of every number: `runs/seeds/robustness_mean_std_top1.md` (5 seeds, mean ± std) and `runs/keypoints_resnet50_256x192/final_metrics.json`.
**Raw-frame CNN column currently has 4 of 5 seeds.** Seed 4 finishes ~01:05 on 2026-11-04. Refresh with
`.venv\Scripts\python models\aggregate_seeds.py` and update the Raw-frame numbers on slides 8-10 (they are marked ◆).

Held back for the final presentation (not in this deck): MediaPipe / RTMPose comparison, the NPU run, final report numbers.

---

## Slide 1: Title
Keypoint extraction for sign language recognition: midsemester update
Group 2, DATA 255, SJSU Fall 2026. Siva Surya Chandran, Sharan Lourduraj, Timothy Song, Rohil Utture, Sri Prajeet Villyal

## Slide 2: Where we started and what changed
**Proposal:** learn a keypoint extractor, pool keypoints over a few frames, classify the ASL word; compare against a raw-frame CNN.
**What changed:**
- Added a **robustness question**: how do keypoints vs. pixels hold up under occlusion, motion blur and missing frames?
- Added **Stage 2**, a temporal refiner that repairs low-confidence keypoints.
- Classifier is a small bi-GRU over 16 frames (the proposal planned plain frame pooling).
**Still true:** from-scratch keypoint head trained on COCO-WholeBody, evaluated against a raw-frame CNN baseline. We are not chasing state-of-the-art accuracy.

## Slide 3: Pipeline
Video → 16 sampled frames → **Stage 1** keypoint extractor (73 points: upper body, mouth, both hands) → **Stage 2** temporal refiner (repairs points with score < 0.2) → **Stage 3** bi-GRU classifier → one of 100 ASL words.
Baseline: ImageNet ResNet-18 on every frame + the same bi-GRU head.

## Slide 4: Data
- **WLASL100:** 2,038 clips, 100 words, official splits: 1,442 train / 338 val / 258 test.
- Frames: 16 per clip at 224×224 (the final data comes from a 256×256 letterboxed copy of WLASL; black bars cropped, aspect ratio kept).
- **COCO-WholeBody:** supervises the keypoint extractor.
- Keypoint layout: 73 points = 11 upper body + 20 mouth + 21 per hand.
- Known limitation: official splits are not signer-disjoint (52 of 56 test signers also appear in train).
- WLASL is C-UDA licensed; clips are kept private.

## Slide 5: Stage 1, the keypoint extractor
- ResNet-50 backbone (**ImageNet-initialised**) + 3 deconvolution layers → 73 heatmaps; keypoint head trained on COCO-WholeBody, 140 epochs, 256×192 input.
- **COCO val, PCK@0.1:** 95.2% overall; upper body 95.1%, mouth 99.2%, left hand 93.6%, right hand 93.3%. PCK@0.05: 90.4%. NME 0.030.
- **Domain gap check:** 100 hand-labeled WLASL frames (90 scored), error as a fraction of shoulder width: PCK@0.2 = 97.6%, PCK@0.1 = 71.7%.
- Visual check: keypoint overlays on WLASL frames (use the `overlay_*.png` files in `runs/pipeline/keypoints/clean/`).

## Slide 6: Testing robustness: three degradations × three severities
Applied to the frames before Stage 1 (identical damage for every model: seeded per clip).
| Type | What it does | mild / moderate / severe |
|---|---|---|
| Occlusion | grey square over one hand | 25% / 50% / 75% of frames; square side 20% / 25% / 30% of frame |
| Motion blur | random-angle blur | 25% / 50% / 75% of frames; kernel 7 / 13 / 21 px |
| Missing frames | frames replaced by black | 25% / 50% / 75% of frames |

10 conditions in total (clean + 9). Metrics on the 258 test clips: top-1, top-5, macro-F1.

## Slide 7: What we compare (all trained on clean data only)
- **Frame-only:** Stage 3 on Stage 1 keypoints.
- **+Interpolation:** replaces low-confidence points by interpolation (no training; after Roh et al. 2024).
- **+Temporal:** Stage 2 refiner trained on synthetic keypoint noise.
- **Proposed (+Temporal & robust training):** Stage 2 refiner trained on keypoints from really degraded frames.
- **Raw-frame CNN:** ResNet-18 + bi-GRU on pixels.
5 seeds each for Stages 2-3 and the raw-frame model (Stage 1 is one trained checkpoint). Mean ± std.

## Slide 8: Results: top-1 accuracy (%), mean ± std over seeds
| | Frame-only | +Interp. | +Temporal | **Proposed** | Raw-frame CNN ◆ |
|---|---|---|---|---|---|
| clean | 49.5 ± 1.3 | 46.7 ± 1.5 | 46.4 ± 1.5 | 48.8 ± 2.6 | 38.6 ± 1.5 |
| occlusion severe | 31.7 ± 0.8 | 31.7 ± 2.2 | 29.8 ± 1.2 | **36.4 ± 1.3** | 22.7 ± 3.1 |
| blur severe | 31.5 ± 2.1 | 29.8 ± 1.4 | 32.6 ± 0.8 | **40.0 ± 1.2** | 17.6 ± 3.5 |
| missing moderate | 31.7 ± 3.3 | 36.4 ± 1.4 | 28.2 ± 4.5 | **42.8 ± 1.9** | 32.4 ± 1.2 |
| missing severe | 13.8 ± 3.1 | 25.0 ± 2.0 | 12.4 ± 2.5 | **29.5 ± 2.2** | 20.3 ± 1.0 |
| **avg of 9 degraded** | 36.6 ± 1.3 | 37.5 ± 1.4 | 34.6 ± 1.0 | **42.2 ± 1.7** | 29.7 ± 1.3 |
| drop, clean → degraded | −12.9 | −9.2 | −11.8 | **−6.6** | −8.9 |
Chance level is 1% (100 classes). Full 10-condition table: `runs/seeds/robustness_mean_std_top1.md`.

## Slide 9: What the results say
- **Proposed** is the most robust: it loses 6.6 points under degradation, Frame-only loses 12.9.
- Its lead is clearest at **severe blur, severe occlusion and every missing-frame level**.
- On clean clips and at **mild** occlusion or blur it is level with Frame-only (differences within one std). We do not claim a win there.
- **+Interpolation** only helps with missing frames, and costs about 3 points on clean clips.
- **+Temporal** (synthetic noise) does not help: training the refiner on real degradations is what matters.
- **Pixels vs keypoints:** the raw-frame CNN is lower in absolute accuracy (clean 38.6% vs 49.5%) but its drop under degradation (−8.9) is no larger than Frame-only's (−12.9). Only Proposed is clearly more robust than pixels. ◆

## Slide 10: Error analysis: confusion matrix (clean test, n = 258)
Figures: `report/figures/confusion_matrix_proposed.png`, `report/figures/confusion_matrix_frame_only.png`.
- Proposed 50.0% top-1; 17 words with ≥ 2 test clips are all correct, 19 are never correct.
- Most common mistakes (Proposed): candy → hearing (3), thanksgiving → tell (2), dance → study (2), computer → work (2).
- Errors are spread out, not concentrated in a few classes; each class has only ~2-4 test clips.

## Slide 11: Limitations
- Single Stage 1 checkpoint; seeds vary Stages 2-3 and the raw-frame CNN only.
- Test set is small (258 clips, so 1 clip ≈ 0.4 points); one dataset (WLASL100).
- Official splits are not signer-disjoint (98% of test clips come from signers also seen in train).
- Degradations are synthetic (grey squares, linear blur, black frames), applied to a fraction of frames.
- The from-scratch claim applies to the keypoint head; the backbone starts from ImageNet (as does the baseline).

## Slide 12: Remaining before the final presentation
1. **Comparison against prebuilt keypoint extractors** (MediaPipe, RTMPose): in the robustness table and on COCO-WholeBody val.
2. **NPU run:** retrain Stages 2-3 for the Mobilint NPU (waiting on a lab run).
3. **Final report and slides:** regenerate every number from `robustness.csv`, write the results section.

## Speaker notes: likely questions
- *Why not signer-disjoint splits?* We used the official WLASL splits so results compare with the literature. A signer-disjoint split is harder and would be a follow-up.
- *Is Stage 1 really from scratch?* The deconv keypoint head is; the ResNet-50 starts from ImageNet weights.
- *Why does +Temporal not help?* It learned to undo synthetic keypoint noise, which does not look like keypoints from really blurred, occluded or missing frames.
- *Why is accuracy only ~50%?* 100 classes, ~14 training clips per class, keypoints only (no appearance), from-scratch training. This project measures robustness, not state-of-the-art accuracy.
