# Presentation script: Group 2 progress checkup (7-minute limit)

Five speakers, in the order of the names on the title slide. Swap names freely: nothing below depends on who did which part of the project.
Lean version, about 750 words: roughly 5:20 at a normal pace of 140 words per minute, plus a few seconds for each handoff, so about 5:35 in total (6:00 if you speak slowly, 5:00 if quickly). The limit is 7:00. The handoff lines are in italics.

## Speaker 1 · slides 1-2 · about 50 s

**[Slide 1]** Hi everyone, we are Group 2. Our project is keypoint extraction for sign language recognition.

**[Slide 2]** The domain is computer vision for accessibility, and specifically word-level ASL recognition: given a short video clip, which of 100 words is being signed?

Three things make this hard. First, video is high-dimensional, and the meaning is spread across the hands, the mouth and the body together. Second, similar signs differ only in small details, so a single frame is often not enough to tell them apart. Third, real video is imperfect: hands get blocked, motion blurs the image, and frames get dropped.

So our target is a system that stays accurate when the video is imperfect. *Next, existing work.*

## Speaker 2 · slides 3-4 · about 75 s

**[Slide 3]** Existing work falls into two families. Appearance-based models, like I3D and VGG-GRU, learn directly from raw frames. Pose-based models, like Pose-GRU and Pose-TGCN, classify keypoints instead: 55 body and hand points from OpenPose. OpenPose and MediaPipe are typically used as fixed, prebuilt extractors.

That leaves three gaps. The extractor is not a learned part of the pipeline. The pose baselines use hands and upper body but no mouth shape. And results are reported on clean video.

**[Slide 4]** For data, we use WLASL100: 100 words, 2,038 videos from 97 signers, with the official train, validation and test splits. COCO-WholeBody gives us whole-body keypoint labels to train our extractor. For words we report top-1, top-5 and macro-F1, and for keypoints we report PCK and NME.

This chart shows published top-1 accuracy on WLASL100 from the WLASL paper: VGG-GRU about 26 percent, Pose-GRU 46.5, Pose-TGCN 55.4 and I3D 65.9. Our keypoint model, in orange, scores about 49.5 on clean clips, in the range of the pose baselines. The comparison is indicative, since training setups differ. *Now, our approach.*

## Speaker 3 · slides 5-6 · about 75 s

**[Slide 5]** Our pipeline has three stages. A video is sampled to 16 frames. Stage 1 is our own keypoint extractor: a ResNet-50 with a deconvolution head that predicts 73 points covering the upper body, the mouth and both hands. Stage 2 is a temporal refiner that repairs low-confidence points using neighbouring frames. Stage 3 is a bi-GRU classifier.

Progress on Stage 1: 95.2 percent PCK at 0.1 on COCO validation, and 97.6 percent PCK at 0.2 on 100 WLASL frames that we labeled by hand. As a baseline, we run ResNet-18 on raw pixels with the same GRU head.

**[Slide 6]** So what is new compared with prior work? We use a learned extractor, trained on COCO-WholeBody, that includes the mouth. We add a stress test: occlusion, blur and missing frames at three severities. Our Stage 2 refiner is trained on keypoints from really degraded frames. And we run the raw-frame baseline under identical damage.

One honesty note: the keypoint head is trained from scratch, but the backbone starts from ImageNet weights, like the baseline. *Next, how we tested it.*

## Speaker 4 · slides 7-8 · about 70 s

**[Slide 7]** To test robustness, we damage the frames before Stage 1. Occlusion puts a grey square over one hand. Blur applies random-angle motion blur. Missing frames replaces frames with black. Each comes at mild, moderate and severe, hitting 25, 50 or 75 percent of frames, and every model sees exactly the same damage.

We compare five models. Frame-only. Plus interpolation. Plus temporal, with a refiner trained on synthetic noise. Our Proposed model, with a refiner trained on really degraded frames. And the raw-frame CNN. All are trained on clean data only, with five seeds each.

**[Slide 8]** Here is top-1 accuracy, averaged over the five seeds. On clean clips, Frame-only and Proposed are level, around 49. Under heavy damage, Proposed pulls ahead: at severe blur, 40 versus 31.5; at severe occlusion, 36 versus 32; and with severe missing frames, 29.5 versus just 14.

On the right: from clean to the degraded average, Proposed loses 6.6 points, the raw-frame CNN loses 9.1, and Frame-only loses 12.9. *Over to the takeaways.*

## Speaker 5 · slide 9 · about 60 s

**[Slide 9]** Our takeaways. Proposed is the most robust, and its gains grow with the damage. It does not win on clean or mild degradation: there it is within one standard deviation of Frame-only, so we do not claim a win. A refiner trained on synthetic noise does not help, so training on real degradations is what matters. And the raw-frame CNN is lower overall, but its drop is smaller than Frame-only's, so pixels are not clearly more fragile than plain keypoints.

Still to do before the final: compare against MediaPipe and RTMPose, run on the Mobilint NPU, and write the final report. Our limitations, such as a single Stage 1 model, a small test set of 258 clips, and signers overlapping across splits, are in the backup slides.

Thank you. We are happy to take questions.

---

## If you are asked

- **Why only about 50 percent accuracy?** 100 classes, roughly 14 training clips per class, keypoints only with no appearance, and a lightweight model trained from scratch. This project is about robustness, not state-of-the-art accuracy.
- **Is Stage 1 really from scratch?** The keypoint head is. The ResNet-50 backbone starts from ImageNet weights.
- **Why not signer-disjoint splits?** We used the official WLASL splits so results compare with the literature. 98 percent of test clips come from signers also seen in training. A signer-disjoint split would be a follow-up.
- **Why does the synthetic-noise refiner not help?** It learns to undo synthetic keypoint noise, which does not resemble keypoints from really blurred, occluded or missing frames.
- **How big are the differences?** One test clip is about 0.4 points, so differences of a point or two are noise. The gap between Proposed and Frame-only on the degraded average is several standard deviations.
