// Builds report/midsem_deck.pptx from report/results/robustness_mean_std_top1.csv (see report/results/README.md to regenerate it).
// 7-minute talk: 9 main slides (~45 s each) following the course rubric, then backup slides.
//   node report/build_midsem_deck.js     (NODE_PATH must contain pptxgenjs and jszip: runs/tools/node_modules)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = "C:/Users/018362708/AppData/Roaming/Claude/local-agent-mode-sessions/skills-plugin/49c0dff6-0780-4ed6-8975-28802634a0a9/9eb4af96-f27f-46f2-8fe2-a5723bba384a/skills/pptx";
const { applyTheme } = require(SKILL + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(__dirname, "midsem_deck.pptx");

// ---------- data ----------
const csv = fs.readFileSync(path.join(ROOT, "report/results/robustness_mean_std_top1.csv"), "utf8").trim().split(/\r?\n/).slice(1);
const D = {};
for (const line of csv) {
  const [cond, cfg, mean, std, n] = line.split(",");
  D[`${cond}|${cfg}`] = { mean: parseFloat(mean), std: parseFloat(std), n: parseInt(n, 10) };
}
const CFG = ["Frame-only", "+Interpolation", "+Temporal", "+Temporal & robust training", "Raw-frame CNN"];
const g = (cond, cfg) => D[`${cond}|${cfg}`];
const fmt = (x) => `${x.mean.toFixed(1)} ± ${x.std.toFixed(1)}`;
const drop = (cfg) => g("avg_9_degraded", cfg).mean - g("clean", cfg).mean;
const rawN = g("clean", "Raw-frame CNN").n;
const PROPOSED = "+Temporal & robust training";

// ---------- theme ----------
const THEME = {
  name: "Signal Teal",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14262B", lt1: "FFFFFF", dk2: "1F6F78", lt2: "EDF3F3",
    accent1: "1F6F78", accent2: "F2A541", accent3: "D1603D", accent4: "6C8EA0", accent5: "8FB996", accent6: "9A8C98",
    hlink: "1F6F78", folHlink: "6C8EA0",
  },
};
const H = THEME.colors; // hex values for the few options that cannot take scheme colors

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Keypoint extraction for sign language recognition: progress checkup";
pres.author = "Group 2, DATA 255";
pres.subject = "Project progress checkup";
const C = pres.SchemeColor;
const S = pres.shapes;

const FOOT = { x: 0.5, y: 5.27, w: 5, h: 0.25, fontSize: 10, color: C.accent4, isTextBox: true, margin: 0 };
function frame(name, dark) {
  pres.defineSlideMaster({
    title: name,
    background: { color: dark ? C.text1 : C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.8, fontSize: 32, bold: true, color: dark ? C.background1 : C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "DATA 255 · Group 2 · Progress checkup", options: FOOT } },
    ],
    slideNumber: { x: 9.0, y: 5.27, w: 0.5, h: 0.25, fontSize: 10, color: C.accent4, align: "right" },
  });
}
frame("LIGHT", false);
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.25, w: 5.9, h: 1.9, fontSize: 34, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 3.3, w: 5.9, h: 0.5, fontSize: 20, color: C.accent2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});

pres.addSection({ title: "Progress checkup" });
const SEC = { sectionTitle: "Progress checkup" };
const shadow = () => ({ type: "outer", color: H.dk1, blur: 6, offset: 2, angle: 90, opacity: 0.12 });
const card = (s, x, y, w, h, fill, name) =>
  s.addShape(S.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.08, shadow: shadow(), objectName: name });
const badge = (s, x, y, label, fill, txt, name) => {
  s.addShape(S.OVAL, { x, y, w: 0.44, h: 0.44, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: name });
  s.addText(String(label), { x, y, w: 0.44, h: 0.44, align: "center", valign: "middle", fontSize: 16, bold: true, color: txt, margin: 0, isTextBox: true, objectName: name + " number" });
};
const seg = (s, x1, y1, x2, y2, color, name) =>
  s.addShape(S.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), flipV: (x2 - x1) * (y2 - y1) < 0, line: { color, width: 1.75 }, objectName: name });
const th = (t, o = {}) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 14, ...o } });
const td = (t, o = {}) => ({ text: t, options: { color: C.text1, fontSize: 14, ...o } });
const note = (s, text, y = 4.75, name = "Note") =>
  s.addText(text, { x: 0.5, y, w: 9, h: 0.4, fontSize: 12, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: name });
const stat = (s, x, y, w, h, value, label, fill, name, valueColor) => {
  card(s, x, y, w, h, fill, name);
  s.addText(value, { x: x + 0.2, y: y + 0.08, w: w - 0.4, h: 0.7, fontSize: 36, bold: true, color: valueColor || C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: name + " value" });
  s.addText(label, { x: x + 0.2, y: y + 0.82, w: w - 0.4, h: h - 0.9, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: name + " label" });
};

// ======================= MAIN SLIDES =======================

// ===== 1. Title =====
function titleSlide() {
  const s = pres.addSlide({ masterName: "TITLE", ...SEC });
  s.addText("Keypoint extraction for sign language recognition", { placeholder: "title" });
  s.addText("Project progress checkup", { placeholder: "subtitle" });
  s.addText("Group 2 · DATA 255 · SJSU Fall 2026", { x: 0.6, y: 3.95, w: 5.9, h: 0.3, fontSize: 14, color: C.background2, margin: 0, isTextBox: true, objectName: "Course" });
  s.addText("Siva Surya Chandran · Sharan Lourduraj · Timothy Song · Rohil Utture · Sri Prajeet Villyal", { x: 0.6, y: 4.3, w: 5.9, h: 0.55, fontSize: 12, color: C.accent4, margin: 0, isTextBox: true, objectName: "Team" });
  const W = [8.05, 4.55];
  const chains = [
    [[7.45, 4.05], [7.05, 3.7], [6.85, 3.35], [6.8, 3.0]],
    [[7.7, 3.4], [7.55, 2.9], [7.45, 2.5], [7.4, 2.1]],
    [[8.1, 3.3], [8.1, 2.7], [8.1, 2.2], [8.1, 1.8]],
    [[8.5, 3.4], [8.55, 2.85], [8.65, 2.4], [8.7, 2.05]],
    [[8.85, 3.7], [9.0, 3.2], [9.15, 2.85], [9.25, 2.55]],
  ];
  chains.forEach((ch, i) => {
    seg(s, W[0], W[1], ch[0][0], ch[0][1], H.accent4, `bone wrist ${i}`);
    for (let k = 0; k < ch.length - 1; k++) seg(s, ch[k][0], ch[k][1], ch[k + 1][0], ch[k + 1][1], H.accent4, `bone ${i}.${k}`);
  });
  chains.forEach((ch, i) => ch.forEach(([x, y], k) =>
    s.addShape(S.OVAL, { x: x - 0.07, y: y - 0.07, w: 0.14, h: 0.14, fill: { color: k === ch.length - 1 ? C.accent2 : C.accent1 }, line: { color: C.text1, width: 0.75 }, objectName: `joint ${i}.${k}` })));
  s.addShape(S.OVAL, { x: W[0] - 0.09, y: W[1] - 0.09, w: 0.18, h: 0.18, fill: { color: C.accent2 }, line: { color: C.text1, width: 0.75 }, objectName: "wrist joint" });
  s.addNotes("(~10 s) Progress checkup for Group 2. Flow: the problem, existing work and gaps, datasets and metrics, our approach and what is new, then progress and what remains. 7 minutes in total.");
}

// ===== 2. Domain and problem (rubric 1) =====
function domainSlide() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("Domain and problem", { placeholder: "title" });
  card(s, 0.5, 1.3, 3.7, 3.7, C.text2, "Task card");
  s.addText("Word-level sign language recognition", { x: 0.75, y: 1.5, w: 3.2, h: 1.3, fontSize: 24, bold: true, color: C.background1, margin: 0, valign: "top", isTextBox: true, objectName: "Task title" });
  s.addText("Computer vision for accessibility: classify a short ASL clip as one of 100 words.", { x: 0.75, y: 2.85, w: 3.2, h: 1.0, fontSize: 14, color: C.background2, margin: 0, valign: "top", isTextBox: true, objectName: "Task text" });
  s.addText("Our target: stay accurate when the video is imperfect.", { x: 0.75, y: 4.0, w: 3.2, h: 0.85, fontSize: 16, bold: true, color: C.accent2, margin: 0, valign: "top", isTextBox: true, objectName: "Target" });
  const rows = [
    ["Video is high-dimensional.", "Meaning sits in the hands, the mouth and the body together."],
    ["Similar signs differ in small details.", "A single frame is often not enough to tell them apart."],
    ["Real video is imperfect.", "Hands get occluded, motion blurs and frames drop."],
  ];
  rows.forEach(([lead, rest], i) => {
    const y = 1.3 + i * 1.25;
    card(s, 4.45, y, 5.05, 1.1, C.background2, `Challenge ${i + 1}`);
    badge(s, 4.62, y + 0.33, i + 1, C.accent1, C.background1, `Challenge ${i + 1} badge`);
    s.addText([{ text: lead, options: { bold: true, fontSize: 16, color: C.text1, breakLine: true } }, { text: rest, options: { fontSize: 14, color: C.text1 } }],
      { x: 5.25, y: y + 0.08, w: 4.15, h: 0.94, valign: "middle", margin: 0, isTextBox: true, objectName: `Challenge ${i + 1} text` });
  });
  s.addNotes("(~40 s) The domain is video understanding for accessibility: recognising isolated ASL words. Three challenges: high-dimensional video, signs that differ in small hand and mouth details across frames, and real-world video quality (occlusion, blur, dropped frames). Our goal is a pipeline that stays accurate when video is imperfect.");
}

// ===== 3. Existing work and gaps (rubric 2) =====
function existingSlide() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("Existing work and gaps", { placeholder: "title" });
  card(s, 0.5, 1.3, 4.4, 3.7, C.background2, "Existing card");
  s.addText("What exists", { x: 0.75, y: 1.4, w: 3.9, h: 0.4, fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "Existing heading" });
  const b = (lead, rest, last) => [{ text: lead, options: { bold: true, breakLine: false } }, { text: rest, options: { breakLine: !last } }];
  s.addText([
    ...b("Appearance-based: ", "I3D and VGG-GRU learn from raw frames."),
    ...b("Pose-based: ", "Pose-GRU and Pose-TGCN take 55 OpenPose body and hand points."),
    ...b("Prebuilt extractors: ", "OpenPose and MediaPipe, used as fixed tools.", true),
  ], { x: 0.75, y: 1.95, w: 3.9, h: 2.95, fontSize: 16, color: C.text1, valign: "top", paraSpaceAfter: 16, margin: 0, isTextBox: true, objectName: "Existing list" });
  card(s, 5.1, 1.3, 4.4, 3.7, C.accent2, "Gaps card");
  s.addText("Gaps we target", { x: 5.35, y: 1.4, w: 3.9, h: 0.4, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "Gaps heading" });
  const gaps = ["The keypoint extractor is a fixed, prebuilt tool, not a learned part of the pipeline.", "Pose baselines use no mouth shape.", "Results are reported on clean video."];
  gaps.forEach((t, i) => {
    const y = 1.95 + i * 1.0;
    badge(s, 5.35, y + 0.08, i + 1, C.text1, C.background1, `Gap ${i + 1} badge`);
    s.addText(t, { x: 5.95, y, w: 3.4, h: 0.85, fontSize: 14, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `Gap ${i + 1} text` });
  });
  s.addNotes("(~50 s) Source: Li et al., WLASL (WACV 2020). Appearance-based baselines are I3D and VGG-GRU; pose-based baselines are Pose-GRU and Pose-TGCN, which use 55 OpenPose keypoints (13 upper-body joints plus 21 per hand). OpenPose and MediaPipe are widely used prebuilt extractors. The gaps are our framing of what these baselines leave open: a fixed extractor, no mouth shape in the pose input, and results on clean video.");
}

// ===== 4. Datasets, metrics, baseline performance (rubric 3) =====
function datasetsSlide() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("Datasets and metrics", { placeholder: "title" });
  const head = (t, y, name) => s.addText(t, { x: 0.5, y, w: 4.3, h: 0.35, fontSize: 18, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: name });
  head("Datasets", 1.3, "Datasets heading");
  s.addText([
    { text: "WLASL100: ", options: { bold: true } }, { text: "100 words, 2,038 videos, 97 signers. Official train, val and test splits.", options: { breakLine: true } },
    { text: "COCO-WholeBody: ", options: { bold: true } }, { text: "whole-body keypoints that train Stage 1.", options: {} },
  ], { x: 0.5, y: 1.7, w: 4.3, h: 1.55, fontSize: 14, color: C.text1, valign: "top", paraSpaceAfter: 8, margin: 0, isTextBox: true, objectName: "Datasets list" });
  head("Metrics", 3.35, "Metrics heading");
  s.addText([
    { text: "Words: ", options: { bold: true } }, { text: "top-1, top-5, macro-F1.", options: { breakLine: true } },
    { text: "Keypoints: ", options: { bold: true } }, { text: "PCK and NME.", options: {} },
  ], { x: 0.5, y: 3.75, w: 4.3, h: 1.0, fontSize: 14, color: C.text1, valign: "top", paraSpaceAfter: 8, margin: 0, isTextBox: true, objectName: "Metrics list" });
  s.addText("Top-1 accuracy on WLASL100 (%)", { x: 5.1, y: 1.3, w: 4.4, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "Chart title" });
  s.addChart(pres.charts.BAR, [{
    name: "Top-1",
    labels: ["VGG-GRU", "Pose-GRU", "Pose-TGCN", "I3D", "Ours: Frame-only", "Ours: Proposed"],
    values: [25.97, 46.51, 55.43, 65.89, +g("clean", "Frame-only").mean.toFixed(1), +g("clean", PROPOSED).mean.toFixed(1)],
  }], {
    x: 5.0, y: 1.65, w: 4.5, h: 3.1, barDir: "bar", catAxisOrientation: "maxMin", chartColors: [H.accent4, H.accent4, H.accent4, H.accent4, H.accent2, H.accent2],
    showLegend: false, showTitle: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 12, dataLabelColor: H.dk1, dataLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, catAxisLabelColor: H.dk1, catAxisLabelFontFace: "+mn-lt", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 80,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, barGapWidthPct: 45,
  });
  note(s, "Published: Li et al., WACV 2020 (grey). Ours: mean of 5 seeds on clean test clips.", 4.8, "Chart note");
  s.addNotes("(~50 s) Published WLASL100 top-1 / top-5 (Li et al., WACV 2020, Table 3): VGG-GRU 25.97 / 55.04, Pose-GRU 46.51 / 76.74, Pose-TGCN 55.43 / 78.68, I3D 65.89 / 84.11. Our keypoint-only Frame-only model is in the range of the published pose-based baselines. The comparison is indicative: our training setup and keypoint extractor differ. Keypoint metrics: PCK is the fraction of points within a distance threshold; NME is the mean localisation error. WLASL is C-UDA licensed, so clips stay private.");
}

// ===== 5. Approach: pipeline + Stage 1 progress (rubric 4) =====
function pipelineSlide() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("Our approach: the pipeline", { placeholder: "title" });
  const boxes = [
    ["Video", "16 sampled frames", C.background2, C.text1, C.text2],
    ["Stage 1", "Keypoint extractor, 73 points", C.accent1, C.background1, C.background2],
    ["Stage 2", "Temporal refiner fixes weak points", C.accent1, C.background1, C.background2],
    ["Stage 3", "bi-GRU classifier", C.accent1, C.background1, C.background2],
    ["Word", "One of 100 signs", C.background2, C.text1, C.text2],
  ];
  const bw = 1.6, gap = 0.25, x0 = 0.5, y = 1.3, bh = 1.6;
  boxes.forEach(([head, sub, fill, tc, sc], i) => {
    const x = x0 + i * (bw + gap);
    card(s, x, y, bw, bh, fill, `${head} box`);
    s.addText(head, { x: x + 0.12, y: y + 0.12, w: bw - 0.24, h: 0.4, fontSize: 18, bold: true, color: tc, margin: 0, isTextBox: true, objectName: `${head} label` });
    s.addText(sub, { x: x + 0.12, y: y + 0.6, w: bw - 0.24, h: 0.95, fontSize: 14, color: sc, valign: "top", margin: 0, isTextBox: true, objectName: `${head} detail` });
    if (i < boxes.length - 1)
      s.addShape(S.RIGHT_ARROW, { x: x + bw + 0.03, y: y + bh / 2 - 0.17, w: gap - 0.06, h: 0.34, fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 }, objectName: `Arrow ${i + 1}` });
  });
  s.addText("Progress on Stage 1", { x: 0.5, y: 3.1, w: 5, h: 0.3, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "Progress heading" });
  stat(s, 0.5, 3.45, 2.85, 1.5, "95.2%", "PCK@0.1 on COCO val", C.background2, "COCO stat");
  stat(s, 3.55, 3.45, 2.85, 1.5, "97.6%", "PCK@0.2 on 100 hand-labeled WLASL frames", C.background2, "WLASL stat");
  card(s, 6.6, 3.45, 2.9, 1.5, C.accent2, "Baseline card");
  s.addText([{ text: "Raw-frame baseline", options: { bold: true, fontSize: 16, breakLine: true } }, { text: "ImageNet ResNet-18 on pixels, then the same bi-GRU head", options: { fontSize: 14 } }],
    { x: 6.8, y: 3.52, w: 2.5, h: 1.36, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: "Baseline text" });
  s.addNotes("(~60 s) Stage 1: ResNet-50 (ImageNet-initialised) plus three deconvolution layers produces 73 heatmaps (11 upper-body, 20 mouth, 21 per hand); the deconv keypoint head is trained from scratch on COCO-WholeBody, 140 epochs at 256x192. COCO val: PCK@0.1 95.2% overall (upper body 95.1, mouth 99.2, left hand 93.6, right hand 93.3), PCK@0.05 90.4%, NME 0.030. On our 100 hand-labeled WLASL frames (90 scored, error as a fraction of shoulder width) PCK@0.2 is 97.6% and PCK@0.1 is 71.7%. Stage 2 repairs points with confidence below 0.2. Stage 3 is a two-layer bi-GRU over 16 frames. Everything is trained on clean data only.");
}

// ===== 6. What is new (rubric 4) =====
function noveltySlide() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("What is new compared with prior work", { placeholder: "title" });
  s.addTable([
    [th("Prior work"), th("Our approach")],
    [td("Keypoints from a prebuilt extractor"), td("A learned extractor trained on COCO-WholeBody: 73 points, including the mouth", { bold: true })],
    [td("Evaluated on clean video"), td("A stress test: occlusion, blur and missing frames at three severities", { bold: true })],
    [td("Bad keypoints are passed on as they are"), td("Stage 2 refiner trained on keypoints from really degraded frames", { bold: true })],
    [td("Pixels and keypoints rarely compared under the same damage"), td("A raw-frame baseline run under identical damage", { bold: true })],
  ], { x: 0.5, y: 1.3, w: 9, colW: [3.9, 5.1], rowH: [0.45, 0.72, 0.72, 0.72, 0.72], border: { type: "solid", pt: 0.75, color: "D3DCDD" }, valign: "middle", margin: [0, 0.12, 0, 0.12], objectName: "Novelty table" });
  note(s, "Since the proposal: a bi-GRU replaced plain frame pooling, and Stage 2 and the robustness test were added.", 4.85, "Change note");
  s.addNotes("(~50 s) Each row pairs a limitation of existing pipelines with what we do about it. The keypoint head is trained from scratch, but the ResNet-50 backbone starts from ImageNet weights, as does the raw-frame baseline. The proposal planned plain frame pooling; the bi-GRU, Stage 2 and the robustness question were added as the project developed.");
}

// ===== 7. Stress test + models =====
function stressSlide() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("Progress: stress test and models", { placeholder: "title" });
  s.addTable([
    [th("Degradation", { fontSize: 13 }), th("What it does", { fontSize: 13 }), th("Mild · Moderate · Severe", { fontSize: 13 })],
    [td("Occlusion", { bold: true, fontSize: 13 }), td("Grey square over one hand", { fontSize: 13 }), td("Square side 20% · 25% · 30% of frame", { fontSize: 13 })],
    [td("Motion blur", { bold: true, fontSize: 13 }), td("Random-angle linear blur", { fontSize: 13 }), td("Kernel 7 · 13 · 21 px", { fontSize: 13 })],
    [td("Missing frames", { bold: true, fontSize: 13 }), td("Frame replaced by black", { fontSize: 13 }), td("Only the hit frames affected", { fontSize: 13 })],
  ], { x: 0.5, y: 1.25, w: 9, colW: [2.1, 3.1, 3.8], rowH: [0.38, 0.42, 0.42, 0.42], border: { type: "solid", pt: 0.75, color: "D3DCDD" }, valign: "middle", margin: [0, 0.12, 0, 0.12], objectName: "Degradation table" });
  s.addText("Models compared", { x: 0.5, y: 3.15, w: 4, h: 0.3, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "Models heading" });
  const items = [
    ["Frame-only", "Stage 3 on Stage 1 keypoints"],
    ["+ Interpolation", "Fills weak points, no training"],
    ["+ Temporal", "Refiner trained on synthetic noise"],
    ["Proposed", "Refiner trained on real degraded frames"],
    ["Raw-frame CNN", "ResNet-18 on pixels"],
  ];
  const w = 1.7, gap = 0.125;
  items.forEach(([name, desc], i) => {
    const x = 0.5 + i * (w + gap);
    const hot = i === 3;
    card(s, x, 3.5, w, 1.2, hot ? C.accent2 : C.background2, `Config ${i + 1}`);
    s.addText([{ text: name, options: { bold: true, fontSize: 14, breakLine: true } }, { text: desc, options: { fontSize: 12 } }],
      { x: x + 0.1, y: 3.55, w: w - 0.2, h: 1.1, color: C.text1, valign: "middle", margin: 0, isTextBox: true, objectName: `Config ${i + 1} text` });
  });
  note(s, "Identical damage for every model (seeded per clip) · 10 conditions: clean + 9 · 25% / 50% / 75% of frames hit · 5 seeds.", 4.85, "Setup note");
  s.addNotes("(~40 s) Degradations are applied to the frames before Stage 1, with the same seeded damage for every model. Occluders are placed on the clean Stage 1 wrist positions. All models are trained on clean data only; the Proposed refiner is the only one trained on keypoints from really degraded frames. Interpolation follows Roh et al. 2024. 5 seeds vary Stages 2-3 and the raw-frame model; Stage 1 is one checkpoint.");
}

// ===== 8. Results chart =====
function resultsSlide() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("Results: where robustness shows up", { placeholder: "title" });
  const conds = [["Clean", "clean"], ["Occlusion severe", "occlusion_severe"], ["Blur severe", "blur_severe"], ["Missing moderate", "missing_moderate"], ["Missing severe", "missing_severe"]];
  const series = [["Frame-only", "Frame-only"], ["Proposed", PROPOSED], ["Raw-frame CNN", "Raw-frame CNN"]];
  s.addChart(pres.charts.BAR, series.map(([name, cfg]) => ({ name, labels: conds.map((c) => c[0]), values: conds.map((c) => +g(c[1], cfg).mean.toFixed(1)) })), {
    x: 0.4, y: 1.25, w: 6.3, h: 3.85, barDir: "col", barGrouping: "clustered", chartColors: [H.accent4, H.accent2, H.dk2],
    showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: H.dk1, legendFontFace: "+mn-lt",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", dataLabelFontSize: 10, dataLabelColor: H.dk1, dataLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 11, catAxisLabelColor: H.dk1, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelFontSize: 11, valAxisLabelColor: H.dk1, valAxisLabelFontFace: "+mn-lt", valAxisMinVal: 0, valAxisMaxVal: 60, valAxisMajorUnit: 20,
    valGridLine: { color: "D3DCDD", size: 0.5 }, catGridLine: { style: "none" }, barGapWidthPct: 50,
  });
  s.addText("Top-1 drop, clean to degraded average", { x: 7.0, y: 1.3, w: 2.5, h: 0.55, fontSize: 14, bold: true, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "Drop heading" });
  [["Proposed", drop(PROPOSED), C.accent2], ["Raw-frame", drop("Raw-frame CNN"), C.background2], ["Frame-only", drop("Frame-only"), C.background2]].forEach(([name, d, fill], i) => {
    const y = 1.95 + i * 1.05;
    card(s, 7.0, y, 2.5, 0.92, fill, `Drop ${name}`);
    s.addText(d.toFixed(1), { x: 7.15, y: y + 0.05, w: 1.2, h: 0.82, fontSize: 28, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `Drop ${name} value` });
    s.addText(name, { x: 8.3, y: y + 0.05, w: 1.15, h: 0.82, fontSize: 14, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `Drop ${name} label` });
  });
  s.addNotes("(~50 s) Top-1 on the 258 test clips, mean over 5 seeds. The Proposed model loses the least under degradation. At severe blur, severe occlusion and missing frames it leads clearly; on clean clips it is level with Frame-only. Full table with mean and std is on a backup slide.");
}

// ===== 9. Takeaways + what remains =====
function takeawaySlide() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("Takeaways and what remains", { placeholder: "title" });
  card(s, 0.5, 1.3, 5.5, 3.7, C.background2, "Takeaways card");
  s.addText("What the results say", { x: 0.75, y: 1.4, w: 5, h: 0.4, fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true, objectName: "Takeaways heading" });
  const pts = [
    ["Proposed is the most robust: ", `it loses ${Math.abs(drop(PROPOSED)).toFixed(1)} points, Frame\u2011only loses ${Math.abs(drop("Frame-only")).toFixed(1)}.`],
    ["Gains grow with the damage. ", "No win on clean or mild: within one std."],
    ["Real degradations matter: ", "a refiner trained on synthetic noise does not help."],
    ["Pixels vs keypoints: ", `the raw-frame CNN is lower overall, but its drop (${Math.abs(drop("Raw-frame CNN")).toFixed(1)}) is smaller than Frame\u2011only's.`],
  ];
  s.addText(pts.flatMap(([lead, rest], i) => [
    { text: lead, options: { bold: true } },
    { text: rest, options: { breakLine: i < pts.length - 1 } },
  ]), { x: 0.75, y: 1.9, w: 5.0, h: 3.0, fontSize: 14, color: C.text1, valign: "top", paraSpaceAfter: 10, margin: 0, isTextBox: true, objectName: "Takeaways list" });
  card(s, 6.2, 1.3, 3.3, 3.7, C.text2, "Todo card");
  s.addText("Still to do", { x: 6.45, y: 1.4, w: 2.8, h: 0.4, fontSize: 20, bold: true, color: C.background1, margin: 0, isTextBox: true, objectName: "Todo heading" });
  ["Compare with MediaPipe and RTMPose", "NPU run on Mobilint", "Final report and slides"].forEach((t, i) => {
    const y = 2.0 + i * 1.0;
    badge(s, 6.45, y + 0.12, i + 1, C.accent2, C.text1, `Todo ${i + 1} badge`);
    s.addText(t, { x: 7.05, y, w: 2.3, h: 0.8, fontSize: 14, bold: true, color: C.background1, margin: 0, valign: "middle", isTextBox: true, objectName: `Todo ${i + 1} text` });
  });
  s.addNotes("(~55 s) The gap between Proposed and Frame-only on the degraded average is several standard deviations wide. On clean clips and at mild occlusion or blur the differences are within one std, so we do not claim a win there. +Temporal learned to undo synthetic keypoint noise, which does not resemble keypoints from truly blurred, occluded or missing frames. Still to do before the final: the MediaPipe and RTMPose comparison, the Mobilint NPU run (waiting on a lab run), and the final report with all numbers regenerated from robustness.csv. Limitations are on a backup slide.");
}

// ======================= BACKUP SLIDES =======================

function backupTable() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("Backup: full results table", { placeholder: "title" });
  const thh = (t, hot) => ({ text: t, options: { bold: true, fontSize: 12, align: "center", valign: "middle", color: hot ? C.text1 : C.background1, fill: { color: hot ? C.accent2 : C.text2 } } });
  const rows = [
    ["Clean", "clean"], ["Occlusion, severe", "occlusion_severe"], ["Blur, severe", "blur_severe"],
    ["Missing, moderate", "missing_moderate"], ["Missing, severe", "missing_severe"], ["Average, 9 degraded", "avg_9_degraded"],
  ];
  const body = rows.map(([label, cond]) => {
    const best = Math.max(...CFG.map((c) => g(cond, c).mean));
    const avg = cond === "avg_9_degraded";
    return [
      { text: label, options: { bold: true, fontSize: 12, color: C.text1, fill: { color: avg ? C.background2 : C.background1 } } },
      ...CFG.map((c, i) => ({ text: fmt(g(cond, c)), options: { fontSize: 12, align: "center", color: C.text1, bold: g(cond, c).mean === best, fill: { color: i === 3 ? "FDF0DA" : (avg ? C.background2 : C.background1) } } })),
    ];
  });
  const dropRow = [
    { text: "Drop, clean to degraded", options: { bold: true, fontSize: 12, color: C.text1, fill: { color: C.background2 } } },
    ...CFG.map((c, i) => ({ text: drop(c).toFixed(1), options: { fontSize: 12, align: "center", color: C.text1, bold: c === PROPOSED, fill: { color: i === 3 ? "FDF0DA" : C.background2 } } })),
  ];
  s.addTable([[thh("", false), thh("Frame-only"), thh("+ Interp."), thh("+ Temporal"), thh("Proposed", true), thh("Raw-frame CNN")], ...body, dropRow], {
    x: 0.5, y: 1.3, w: 9, colW: [2.2, 1.3, 1.3, 1.3, 1.4, 1.5], rowH: 0.41, valign: "middle", border: { type: "solid", pt: 0.75, color: "D3DCDD" }, margin: [0, 0.08, 0, 0.08], objectName: "Results table",
  });
  note(s, `Top-1 (%), mean ± std over seeds on the 258 test clips; best per row in bold. Raw-frame CNN: ${rawN} of 5 seeds. Chance is 1%.`, 4.85, "Table note");
  s.addNotes("Full 10-condition table: runs/seeds/robustness_mean_std_top1.md. One test clip is about 0.4 points, so differences of a point or two are noise.");
}

function backupLimits() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("Backup: limitations", { placeholder: "title" });
  const items = [
    ["One Stage 1 model", "Seeds vary Stages 2-3 and the raw-frame CNN only."],
    ["Small test set", "258 clips, so one clip is about 0.4 points."],
    ["Signers overlap", "98% of test clips come from signers also seen in train."],
    ["Synthetic damage", "Grey squares, linear blur and black frames."],
    ["Pretrained backbone", "ResNet-50 starts from ImageNet, like the baseline."],
    ["One dataset", "WLASL100 only; no signer-independent split."],
  ];
  items.forEach(([head, rest], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.5 + col * 3.05, y = 1.3 + row * 1.9;
    card(s, x, y, 2.9, 1.75, C.background2, `Limit ${i + 1}`);
    badge(s, x + 0.15, y + 0.14, i + 1, C.accent1, C.background1, `Limit ${i + 1} badge`);
    s.addText(head, { x: x + 0.7, y: y + 0.14, w: 2.1, h: 0.44, fontSize: 16, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `Limit ${i + 1} head` });
    s.addText(rest, { x: x + 0.15, y: y + 0.72, w: 2.6, h: 0.95, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: `Limit ${i + 1} text` });
  });
  s.addNotes("We used the official WLASL splits so results compare with the literature. A signer-disjoint split is harder and would be a follow-up. The deconv keypoint head is trained from scratch; the backbone is ImageNet-initialised.");
}

function backupConfusion() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("Backup: where errors land", { placeholder: "title" });
  s.addImage({ path: path.join(ROOT, "report/results/confusion_matrix_proposed.png"), x: 0.5, y: 1.25, w: 4.4, h: 3.85, altText: "Confusion matrix of the Proposed model on the clean test split: a clear diagonal with scattered single-clip errors" });
  [["50.0%", "top-1 on 258 clean test clips"], ["17 / 19", "words always right / never right (2+ test clips)"], ["Spread out", "no single class dominates the errors"]].forEach(([big, small], i) => {
    const y = 1.3 + i * 1.0;
    card(s, 5.2, y, 4.3, 0.88, C.background2, `Err ${i + 1}`);
    s.addText(big, { x: 5.35, y: y + 0.04, w: 1.75, h: 0.8, fontSize: 24, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: `Err ${i + 1} value` });
    s.addText(small, { x: 7.1, y: y + 0.04, w: 2.3, h: 0.8, fontSize: 13, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `Err ${i + 1} label` });
  });
  s.addText("Common slips: candy → hearing (3×), thanksgiving → tell, dance → study, computer → work (2× each). Only 2 to 4 test clips per word.",
    { x: 5.2, y: 4.35, w: 4.3, h: 0.75, fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true, objectName: "Slips" });
  s.addNotes("Why only about 50%? 100 classes, roughly 14 training clips per class, keypoints only with no appearance, and a from-scratch head. This project measures robustness, not state-of-the-art accuracy.");
}

function backupStage1() {
  const s = pres.addSlide({ masterName: "LIGHT", ...SEC });
  s.addText("Backup: Stage 1 by body group", { placeholder: "title" });
  s.addText("PCK@0.1 on COCO val (%)", { x: 0.5, y: 1.3, w: 5, h: 0.3, fontSize: 14, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "Chart title" });
  s.addChart(pres.charts.BAR, [{ name: "PCK@0.1", labels: ["Upper body", "Mouth", "Left hand", "Right hand"], values: [95.1, 99.2, 93.6, 93.3] }], {
    x: 0.5, y: 1.6, w: 9, h: 3.2, barDir: "col", chartColors: [H.accent1], showLegend: false, showTitle: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 12, dataLabelColor: H.dk1, dataLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, catAxisLabelColor: H.dk1, catAxisLabelFontFace: "+mn-lt", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 110,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, barGapWidthPct: 80,
  });
  note(s, "ResNet-50 (ImageNet init) + 3 deconv layers, 73 heatmaps, 140 epochs, 256×192 input. PCK@0.05 90.4%, NME 0.030.", 4.85, "Stage 1 note");
  s.addNotes("On our hand-labeled WLASL frames (90 of 100 scored, error as a fraction of shoulder width) PCK@0.2 is 97.6% and PCK@0.1 is 71.7%.");
}

titleSlide(); domainSlide(); existingSlide(); datasetsSlide(); pipelineSlide(); noveltySlide(); stressSlide(); resultsSlide(); takeawaySlide();
backupTable(); backupLimits(); backupConfusion(); backupStage1();

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT, "| raw-frame seeds:", rawN);
})();
