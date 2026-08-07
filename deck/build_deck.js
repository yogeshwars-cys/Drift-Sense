/*
 * Builds the Applied Materials hackathon submission deck.
 *
 *   node deck/build_deck.js
 *
 * Every number in FACTS below was produced by running the code in this repo --
 * see the `source` note on each. Update FACTS, not the slide bodies.
 */
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const ROOT = path.join(__dirname, "..");
const OUT = path.join(__dirname, "DriftSense_Submission.pptx");

// ---------------------------------------------------------------- facts ----
const F = {
  // benchmark.py --dataset dataset_primary  (n=100)
  accAll: "49.0%",
  accSolvable: "61.2%",
  accTight: "37.0%",
  medianSolvable: "5.9px",
  runtimeMean: "2.9s",
  runtimeMax: "7.1s",
  // probes/rank_probe.py + primary_results.json, grouped by landmark
  accCorner: "92.5%",
  accGate: "42.1%",
  accVia: "19.0%",
  proposed: "87.5%",
  // primary_results.json vs ground truth (median; p90 is 6.9%), probes/rot_probe.py
  scaleErr: "0.10%",
  rotErr: "0.15°",
  // induction.py / probes/induction_probe.py
  inductionCaught: "13 / 13",
  inductionR: "r = -0.346, p = 0.00043",
  inductionPrec: "84.6%",
  // dataset_generator.py
  nPairs: "300 pairs across 3 splits",
  genTime: "~200s per 100 pairs",
  // probes/reranker_eval.py -- held-out, three disjoint splits
  cnnRank1: "2.9%",
  lmRank1: "38.6%",
  // robustness_sweep.py
  noiseRobust: "no measurable loss at 2x or 3x search-side sensor noise, though "
    + "accuracy does fall away from the tuned lattice-pitch band",
  // train_ranker.py + probes/ranker_report.py, held out on seed 11
  rankerOverall: "64.3% -> 67.1%",
  rankerVia: "16.7% -> 41.7%",
  rankerP: "p = 0.69",
};

// -------------------------------------------------------------- palette ----
const C = {
  ink: "0F141A",      // graphite, dark slide background
  panel: "1A222C",    // raised dark panel
  paper: "FFFFFF",
  wash: "F2F5F7",     // light card fill
  text: "1A222C",
  muted: "63727F",
  line: "D8E0E6",
  cyan: "1FB6C4",     // accent -- inspection-tool phosphor
  cyanLt: "D6F2F5",
  amber: "D98A21",    // limits, failure cases
  amberLt: "FBEEDA",
};

const HFONT = "Cambria";
const BFONT = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5
const W = 13.33, H = 7.5, M = 0.65;

// evenly spaced columns spanning the content width, gap between each
function cols(n, gap) {
  const total = W - M * 2;
  const w = (total - gap * (n - 1)) / n;
  return { w, x: (i) => M + i * (w + gap) };
}

// ------------------------------------------------------------- helpers ----
let slideNo = 0;

function darkSlide() {
  slideNo++;
  const s = pres.addSlide();
  s.background = { color: C.ink };
  return s;
}

// The chip carries the i4C template slide number, so a reviewer can map each
// slide straight onto the required template. It must therefore count the dark
// slides too, even though they don't display it.
function lightSlide(title, kicker) {
  slideNo++;
  const s = pres.addSlide();
  s.background = { color: C.paper };
  // numbered chip -- the repeated motif
  s.addShape(pres.ShapeType.roundRect, {
    x: M, y: 0.44, w: 0.46, h: 0.46, rectRadius: 0.1,
    fill: { color: C.cyanLt }, line: { color: C.cyan, width: 1 },
  });
  s.addText(String(slideNo), {
    x: M, y: 0.44, w: 0.46, h: 0.46, align: "center", valign: "middle",
    fontFace: BFONT, fontSize: 14, bold: true, color: C.cyan, margin: 0,
  });
  s.addText(title, {
    x: M + 0.68, y: 0.38, w: W - M * 2 - 0.68, h: 0.58,
    fontFace: HFONT, fontSize: 30, bold: true, color: C.text, margin: 0, valign: "middle",
  });
  if (kicker) {
    s.addText(kicker, {
      x: M + 0.68, y: 0.96, w: W - M * 2 - 0.68, h: 0.42,
      fontFace: BFONT, fontSize: 12.5, italic: true, color: C.muted, margin: 0,
    });
  }
  return s;
}

// a content card: tinted background, no edge stripes
function card(s, o) {
  s.addShape(pres.ShapeType.roundRect, {
    x: o.x, y: o.y, w: o.w, h: o.h, rectRadius: 0.04,
    fill: { color: o.fill || C.wash },
    line: { color: o.stroke || C.line, width: 1 },
  });
}

function stat(s, x, y, w, value, label, color) {
  s.addText(value, {
    x, y, w, h: 0.62, fontFace: HFONT, fontSize: 38, bold: true,
    color: color || C.cyan, margin: 0, align: "left",
  });
  s.addText(label, {
    x, y: y + 0.62, w, h: 0.5, fontFace: BFONT, fontSize: 11,
    color: C.muted, margin: 0, align: "left",
  });
}

function bullets(s, items, o) {
  s.addText(
    items.map((t, i) => ({
      text: t,
      options: { bullet: true, breakLine: i !== items.length - 1 },
    })),
    {
      x: o.x, y: o.y, w: o.w, h: o.h,
      fontFace: BFONT, fontSize: o.size || 13.5, color: o.color || C.text,
      lineSpacingMultiple: 1.15, paraSpaceAfter: 7, margin: 0, valign: "top",
    }
  );
}

// ============================================================== SLIDE 1 ====
{
  const s = darkSlide();
  s.addText("DRIFT-SENSE", {
    x: M, y: 1.25, w: 9.4, h: 0.95,
    fontFace: HFONT, fontSize: 54, bold: true, color: C.paper,
    charSpacing: 3, margin: 0,
  });
  s.addText("Navigation-Error Recovery for Wafer Inspection", {
    x: M, y: 2.2, w: 9.4, h: 0.55,
    fontFace: BFONT, fontSize: 20, color: C.cyan, margin: 0,
  });
  s.addText(
    "Finding one site inside a sea of identical repeating cells — by measuring the "
    + "geometry instead of searching it.",
    { x: M, y: 2.8, w: 8.6, h: 0.7, fontFace: BFONT, fontSize: 13.5, color: "9BAAB6", margin: 0 }
  );

  // team block -- PLACEHOLDERS, fill before submitting
  card(s, { x: M, y: 3.85, w: 7.4, h: 2.5, fill: C.panel, stroke: "2C3742" });
  s.addText("TEAM", {
    x: M + 0.35, y: 4.05, w: 3, h: 0.3, fontFace: BFONT, fontSize: 10.5,
    bold: true, color: C.cyan, charSpacing: 2, margin: 0,
  });
  const team = [
    ["Team name", "«TEAM NAME»"],
    ["Members & roles", "«NAME — role»,  «NAME — role»,  «NAME — role»"],
    ["College", "«COLLEGE NAME»"],
    ["Contact", "«EMAIL»  ·  «PHONE»"],
  ];
  team.forEach(([k, v], i) => {
    const y = 4.45 + i * 0.44;
    s.addText(k, {
      x: M + 0.35, y, w: 1.85, h: 0.34, fontFace: BFONT, fontSize: 11,
      color: "7D8D9A", margin: 0, valign: "middle",
    });
    s.addText(v, {
      x: M + 2.25, y, w: 5.3, h: 0.34, fontFace: BFONT, fontSize: 12,
      color: C.paper, bold: true, margin: 0, valign: "middle",
    });
  });

  // headline stats
  const sx = 8.45;
  card(s, { x: sx, y: 3.85, w: W - sx - M, h: 2.5, fill: C.panel, stroke: "2C3742" });
  const st = [
    [F.accCorner, "accuracy where a non-periodic\nlandmark is in the field of view"],
    [F.scaleErr, "median error in the recovered\n10x magnification"],
    [F.runtimeMean, "per 1000x1000 image pair,\nsingle CPU core"],
  ];
  st.forEach(([v, l], i) => {
    const y = 4.02 + i * 0.75;
    s.addText(v, {
      x: sx + 0.32, y, w: 1.5, h: 0.5, fontFace: HFONT, fontSize: 25, bold: true,
      color: C.cyan, margin: 0, valign: "middle",
    });
    s.addText(l, {
      x: sx + 1.92, y, w: 2.0, h: 0.6, fontFace: BFONT, fontSize: 9,
      color: "9BAAB6", margin: 0, valign: "middle",
    });
  });

  s.addText("Applied Materials · i4C Hackathon · Problem Statement: Drift-Sense", {
    x: M, y: 6.65, w: 11, h: 0.3, fontFace: BFONT, fontSize: 10, color: "5E6E7A", margin: 0,
  });
  s.addNotes("Replace all «...» placeholders with real team details before submitting.");
}

// ============================================================== SLIDE 2 ====
{
  const s = lightSlide("Problem Statement Addressed",
    "Selected: Drift-Sense — Navigation-Error Recovery");

  s.addText(
    "A wafer inspection tool must return to the same site thousands of times a day. "
    + "Between visits the stage drifts — thermal expansion, fab-floor vibration, mechanical "
    + "slack — and the tool lands several pixels away.",
    { x: M, y: 1.55, w: 6.5, h: 1.0, fontFace: BFONT, fontSize: 14, color: C.text, margin: 0, lineSpacingMultiple: 1.2 }
  );
  s.addText(
    "The tool cannot detect this from the landed image, because every die carries the same "
    + "repeating layout: the structure at the wrong location looks almost identical to the right one.",
    { x: M, y: 2.6, w: 6.5, h: 0.95, fontFace: BFONT, fontSize: 14, color: C.text, margin: 0, lineSpacingMultiple: 1.2 }
  );

  card(s, { x: M, y: 3.7, w: 6.5, h: 1.15, fill: C.cyanLt, stroke: C.cyan });
  s.addText(
    "Why it matters: if the tool cannot prove it is on the right cell, every measurement "
    + "trended over time is comparing two different places on the die.",
    { x: M + 0.28, y: 3.85, w: 6.0, h: 0.9, fontFace: BFONT, fontSize: 13, bold: true, color: "0E5C64", margin: 0, valign: "middle" }
  );

  bullets(s, [
    "Classical template matching breaks down exactly where the layouts are most regular — DRAM arrays, FinFET gate fields.",
    "In a 657x657px window at a 9px pitch, ~30 correlation peaks sit within 0.003 of the global maximum.",
    "Choosing among them is not a matching problem. It is an identifiability problem.",
  ], { x: M, y: 5.05, w: 6.5, h: 1.6, size: 12.5 });

  // right: the ambiguity motif -- a grid of identical cells, one marked
  const gx = 7.55, gy = 1.62, cell = 0.52, n = 9;
  card(s, { x: gx - 0.18, y: gy - 0.18, w: cell * n + 0.36, h: cell * n + 0.36, fill: C.wash, stroke: C.line });
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const isTrue = (r === 5 && c === 3);
      s.addShape(pres.ShapeType.ellipse, {
        x: gx + c * cell + 0.11, y: gy + r * cell + 0.11, w: cell - 0.22, h: cell - 0.22,
        fill: { color: isTrue ? C.cyan : "C3CDD5" }, line: { color: C.paper, width: 1 },
      });
    }
  }
  s.addText("Every cell is a valid-looking match. Only one is correct.", {
    x: gx - 0.18, y: gy + cell * n + 0.3, w: cell * n + 0.36, h: 0.4,
    fontFace: BFONT, fontSize: 11, italic: true, color: C.muted, align: "center", margin: 0,
  });
}

// ============================================================== SLIDE 3 ====
{
  const s = lightSlide("Idea Description",
    "Both architectures · classical, measurement-first localization · zero learned weights on the shipped path");

  const cards = [
    ["Architecture", "Both DRAM and FinFET.",
      "The generator renders word-line/bit-line grids with contact vias, and dense fin fields "
      + "crossed by gate bars. Both are evaluated in every split, 50/50."],
    ["Method", "Classical / geometric — by measurement.",
      "Two learned alternatives were built and evaluated held-out. A CNN re-ranker scores "
      + F.cnnRank1 + " against " + F.lmRank1 + " for the classical matched filter. A listwise "
      + "learned ranker reaches " + F.rankerOverall + " (" + F.rankerP + ") — real, but not significant."],
    ["Core idea", "Measure the geometry; then discriminate.",
      "The unknown ~10x magnification IS the ratio of the two lattice pitches. Measure it, and "
      + "the search collapses from a blind sweep to a handful of geometrically permitted positions."],
  ];
  cards.forEach(([tag, head, body], i) => {
    const x = M + i * 4.06;
    card(s, { x, y: 1.72, w: 3.78, h: 2.5 });
    s.addText(tag.toUpperCase(), {
      x: x + 0.28, y: 1.92, w: 3.2, h: 0.28, fontFace: BFONT, fontSize: 9.5,
      bold: true, color: C.cyan, charSpacing: 1.5, margin: 0,
    });
    s.addText(head, {
      x: x + 0.28, y: 2.24, w: 3.24, h: 0.62, fontFace: HFONT, fontSize: 15,
      bold: true, color: C.text, margin: 0,
    });
    s.addText(body, {
      x: x + 0.28, y: 2.92, w: 3.24, h: 1.2, fontFace: BFONT, fontSize: 11.5,
      color: C.muted, margin: 0, lineSpacingMultiple: 1.12,
    });
  });

  s.addText("Why this beats template matching on periodic layouts", {
    x: M, y: 4.5, w: 8, h: 0.4, fontFace: HFONT, fontSize: 17, bold: true, color: C.text, margin: 0,
  });

  const rows = [
    ["Template matching", "Sweeps a scale band blindly, slides the template, takes the argmax of the correlation surface."],
    ["Drift-Sense", "Measures magnification and rotation from the spectrum, spectrally subtracts the lattice so only "
      + "aperiodic landmarks remain, then makes the surviving candidates compete against each other."],
  ];
  rows.forEach(([k, v], i) => {
    const y = 5.02 + i * 0.86;
    card(s, { x: M, y, w: W - M * 2, h: 0.76, fill: i ? C.cyanLt : C.wash, stroke: i ? C.cyan : C.line });
    s.addText(k, {
      x: M + 0.28, y, w: 2.0, h: 0.76, fontFace: BFONT, fontSize: 12.5, bold: true,
      color: i ? "0E5C64" : C.muted, margin: 0, valign: "middle",
    });
    s.addText(v, {
      x: M + 2.35, y, w: W - M * 2 - 2.65, h: 0.76, fontFace: BFONT, fontSize: 12,
      color: C.text, margin: 0, valign: "middle",
    });
  });
}

// ============================================================== SLIDE 4 ====
{
  const s = lightSlide("Proposed Solution",
    "Dataset generator (left) and localization pipeline (right) — every augmentation cited, see slide 9");

  // ---- left: generator
  card(s, { x: M, y: 1.62, w: 5.5, h: 5.1 });
  s.addText("Dataset generator  ·  dataset_generator.py", {
    x: M + 0.3, y: 1.82, w: 5.0, h: 0.34, fontFace: HFONT, fontSize: 15, bold: true, color: C.text, margin: 0,
  });
  s.addText("--style {dram,finfet,both}  --n  --out  --difficulty-mix  --placement  --noise-scale  --pitch-shift", {
    x: M + 0.3, y: 2.16, w: 5.0, h: 0.5, fontFace: "Courier New", fontSize: 8.5, color: C.muted, margin: 0,
  });
  bullets(s, [
    "Scene rendered supersampled, then area-averaged down 10x — demagnification without aliasing.",
    "Imaging chain in physical order: edge brightening → astigmatic PSF → scan distortion → charging → shading → sensor noise.",
    "Independent Poisson + Gaussian draw per capture; search side uniformly noisier than reference.",
    "Rotation applied to the reference crop only — rotating the search canvas silently corrupts ground truth.",
    "Aperiodic landmarks injected deliberately: array corner, dropped/doubled via, gate crossing.",
    "Ground truth records the true centre AND whether the site is identifiable at all.",
  ], { x: M + 0.3, y: 2.74, w: 5.0, h: 3.8, size: 11.5 });

  // ---- right: pipeline flow
  const px = 6.55, pw = W - px - M;
  card(s, { x: px, y: 1.62, w: pw, h: 5.1 });
  s.addText("Localization  ·  localize.py", {
    x: px + 0.3, y: 1.82, w: pw - 0.6, h: 0.34, fontFace: HFONT, fontSize: 15, bold: true, color: C.text, margin: 0,
  });
  s.addText("python localize.py reference.png search.png   →   \"x, y\"", {
    x: px + 0.3, y: 2.16, w: pw - 0.6, h: 0.3, fontFace: "Courier New", fontSize: 9, color: C.muted, margin: 0,
  });

  const steps = [
    ["Lattice sensor", "2-D spectrum → pitch, orientation. Magnification = pitch ratio, measured to " + F.scaleErr + "."],
    ["Rotation + phase lock", "Sub-bin multi-harmonic angle (" + F.rotErr + "). Cross-image phase pins the centre modulo the pitch."],
    ["Aperiodic residual", "Spectral notch deletes the lattice. Only landmarks survive — the sole channel carrying absolute identity."],
    ["Propose", "Decimated multi-scale NCC over the full frame, plus residual peaks. Cheap, and only ever proposes."],
    ["Rescore at full res", "Candidate crops upsampled to reference resolution — decided where cells actually differ."],
    ["Decide", "Landmark evidence judges. The mandated centre rule breaks genuine ties only."],
  ];
  steps.forEach(([k, v], i) => {
    const y = 2.6 + i * 0.68;
    s.addShape(pres.ShapeType.ellipse, {
      x: px + 0.3, y: y + 0.06, w: 0.28, h: 0.28,
      fill: { color: i >= 4 ? C.cyan : C.cyanLt }, line: { color: C.cyan, width: 1 },
    });
    s.addText(String(i + 1), {
      x: px + 0.3, y: y + 0.06, w: 0.28, h: 0.28, align: "center", valign: "middle",
      fontFace: BFONT, fontSize: 8, bold: true, color: i >= 4 ? C.paper : C.cyan, margin: 0,
    });
    s.addText(k, {
      x: px + 0.68, y, w: pw - 1.0, h: 0.26, fontFace: BFONT, fontSize: 12, bold: true, color: C.text, margin: 0,
    });
    s.addText(v, {
      x: px + 0.68, y: y + 0.25, w: pw - 1.0, h: 0.42, fontFace: BFONT, fontSize: 10, color: C.muted, margin: 0,
    });
  });
}

// ============================================================== SLIDE 5 ====
{
  const s = lightSlide("Innovation & Uniqueness",
    "Four things a stronger backbone would not have given us");

  const items = [
    ["The 10x scale is measured, not searched",
      "The ratio of the two lattice pitches IS the unknown magnification. A blind 8-point sweep over "
      + "8.3–11.7 becomes a tight measured bracket, so the reference footprint lands within ~1px of truth."],
    ["The lattice is spectrally subtracted before matching",
      "Correlating the two aperiodic residuals isolates exactly the content that carries absolute identity. "
      + "This is also what makes honest abstention possible: in a true array interior the channel is flat by construction."],
    ["Candidates compete instead of being scored alone",
      "Beliefs are normalised across the candidate set, and a lattice-phase consensus term scores each candidate "
      + "against the phase the others agree on — information invisible to any per-patch classifier."],
    ["Spatial induction: making the lattice prove its own geometry",
      "The pitch is tested the way one proves a statement over the integers — if pitch p is real, lag n·p must "
      + "also be a peak. Catches " + F.inductionCaught + " silent lattice failures at zero false alarms, and predicts "
      + "solvability (" + F.inductionR + ")."],
  ];
  items.forEach(([h, b], i) => {
    const x = M + (i % 2) * 6.15, y = 1.72 + Math.floor(i / 2) * 2.22;
    card(s, { x, y, w: 5.85, h: 2.02 });
    s.addShape(pres.ShapeType.roundRect, {
      x: x + 0.28, y: y + 0.26, w: 0.34, h: 0.34, rectRadius: 0.1,
      fill: { color: C.cyan }, line: { color: C.cyan, width: 1 },
    });
    s.addText(String(i + 1), {
      x: x + 0.28, y: y + 0.26, w: 0.34, h: 0.34, align: "center", valign: "middle",
      fontFace: BFONT, fontSize: 10, bold: true, color: C.paper, margin: 0,
    });
    s.addText(h, {
      x: x + 0.74, y: y + 0.24, w: 4.85, h: 0.44, fontFace: HFONT, fontSize: 13.5,
      bold: true, color: C.text, margin: 0, valign: "middle",
    });
    s.addText(b, {
      x: x + 0.3, y: y + 0.78, w: 5.28, h: 1.1, fontFace: BFONT, fontSize: 11,
      color: C.muted, margin: 0, lineSpacingMultiple: 1.12,
    });
  });

  card(s, { x: M, y: 6.28, w: W - M * 2, h: 0.72, fill: C.amberLt, stroke: C.amber });
  s.addText(
    "And one meta-choice: every idea measured that did NOT pay is documented rather than deleted — the "
    + "centre-distance prior, the phase penalty, the calibrated commit gate, induction as a scale gate, and the "
    + "CNN re-ranker (held-out: right on 0 trials the classical fusion gets wrong, so the weight sweep picks 0).",
    { x: M + 0.28, y: 6.28, w: W - M * 2 - 0.56, h: 0.72, fontFace: BFONT, fontSize: 11,
      color: "6B4410", margin: 0, valign: "middle" }
  );
}

// ============================================================== SLIDE 6 ====
{
  const s = lightSlide("Results",
    "100 randomized pairs, uniform placement, 50/50 DRAM/FinFET — produced by benchmark.py");

  // ---- headline stats
  const c4 = cols(4, 0.2);
  const stats = [
    [F.accAll, "within 15px\nall 100 pairs", C.text],
    [F.accSolvable, "within 15px\nidentifiable pairs", C.cyan],
    [F.accTight, "within 5px\nall 100 pairs", C.text],
    [F.runtimeMean, "per pair\n1000x1000, 1 CPU core", C.text],
  ];
  stats.forEach(([v, l, col], i) => {
    card(s, { x: c4.x(i), y: 1.62, w: c4.w, h: 1.18 });
    stat(s, c4.x(i) + 0.24, 1.70, c4.w - 0.4, v, l, col);
  });

  // ---- accuracy by landmark: the real finding
  s.addText("Accuracy is set by what is in the field of view — not by architecture", {
    x: M, y: 2.94, w: W - M * 2, h: 0.36,
    fontFace: HFONT, fontSize: 15, bold: true, color: C.text, margin: 0, valign: "middle",
  });
  const lm = [
    ["Array corner", F.accCorner, C.cyan],
    ["Gate crossing", F.accGate, C.muted],
    ["Single via defect", F.accVia, C.amber],
    ["No landmark in FOV", "0%", C.amber],
  ];
  lm.forEach(([k, v, col], i) => {
    const x = c4.x(i);
    card(s, { x, y: 3.34, w: c4.w, h: 0.68 });
    s.addText(v, {
      x: x + 0.24, y: 3.34, w: 0.95, h: 0.68, fontFace: HFONT, fontSize: 19, bold: true,
      color: col, margin: 0, valign: "middle",
    });
    s.addText(k, {
      x: x + 1.22, y: 3.34, w: c4.w - 1.46, h: 0.68, fontFace: BFONT, fontSize: 10.5,
      color: C.muted, margin: 0, valign: "middle",
    });
  });

  // ---- the two visual examples, side by side
  const c2 = cols(2, 0.35);
  const IMG_AR = 0.447; // examples are ~1341 x 600
  const iw = c2.w, ih = iw * IMG_AR;
  const ex = [
    ["examples/success_case.png", "SUCCESS — array corner in FOV, error 0.4px", C.cyan],
    ["examples/failure_case.png",
      "HONEST FAILURE — defect-free array interior, error 320.5px, and confidence 0.85: wrong AND sure.", C.amber],
  ];
  ex.forEach(([rel, cap, col], i) => {
    const x = c2.x(i);
    s.addText(cap, {
      x, y: 4.14, w: iw, h: 0.42, fontFace: BFONT, fontSize: 10.5, bold: true,
      color: col, margin: 0, valign: "middle",
    });
    const p = path.join(ROOT, rel);
    if (fs.existsSync(p)) s.addImage({ path: p, x, y: 4.6, w: iw, h: ih });
  });
  s.addNotes(
    "A DRAM reference footprint spans ~78 lattice cells and a dropped via alters exactly one of them -- "
    + "roughly 1.3% of the content separating the true site from its lattice-shifted rival. "
    + F.proposed + " of identifiable sites ARE correctly proposed by the candidate stage; the loss is in "
    + "ranking them, not in finding them."
  );
}

// ============================================================== SLIDE 7 ====
{
  const s = lightSlide("Technology & Feasibility",
    "Torch-free inference — a reviewer installs four packages and runs one command");

  const specs = [
    ["Language / stack", "Python 3.12 · NumPy · OpenCV · SciPy · Pillow. No GPU, no learned weights on the shipped path."],
    ["Hardware used", "Development and every benchmark on a single consumer CPU core. No cloud, no accelerator."],
    ["Dataset generation", F.genTime + " for a 100-pair split (1000x1000 reference + 1000x1000 search, supersampled render)."],
    ["Inference per pair", F.runtimeMean + " mean, " + F.runtimeMax + " worst case, for a 1000x1000 pair on one CPU core."],
    ["Model size", "0 bytes. The shipped path carries no weights. The unvalidated research re-ranker is a 32-d Siamese CNN (136 KB)."],
    ["Reproducibility", "requirements.txt is the complete pip freeze; requirements-inference.txt is the 4-package torch-free subset."],
  ];
  specs.forEach(([k, v], i) => {
    const y = 1.66 + i * 0.78;
    card(s, { x: M, y, w: W - M * 2, h: 0.68 });
    s.addText(k, {
      x: M + 0.3, y, w: 2.6, h: 0.68, fontFace: BFONT, fontSize: 12, bold: true, color: C.cyan, margin: 0, valign: "middle",
    });
    s.addText(v, {
      x: M + 3.05, y, w: W - M * 2 - 3.35, h: 0.68, fontFace: BFONT, fontSize: 11.5, color: C.text, margin: 0, valign: "middle",
    });
  });

  s.addText(
    "Robustness measured, not assumed: " + F.noiseRobust
    + " — the spectral front end does not degrade the way a pixel-domain matcher would.",
    { x: M, y: 6.42, w: W - M * 2, h: 0.5, fontFace: BFONT, fontSize: 11.5, italic: true, color: C.muted, margin: 0 }
  );
}

// ============================================================== SLIDE 8 ====
{
  const s = darkSlide();
  s.addText("Repository & Demo", {
    x: M, y: 1.5, w: 10, h: 0.8, fontFace: HFONT, fontSize: 38, bold: true, color: C.paper, margin: 0,
  });

  const links = [
    ["GITHUB  (mandatory)", "«https://github.com/«USER»/drift-sense»", "Public repository — generator, inference script, benchmarks, probes, citations."],
    ["VIDEO  (optional)", "«LINK — algorithm running on one sample pair»", "Recommended: show localize.py run end-to-end on a fresh pair."],
  ];
  links.forEach(([k, v, note], i) => {
    const y = 2.6 + i * 1.5;
    card(s, { x: M, y, w: W - M * 2, h: 1.25, fill: C.panel, stroke: "2C3742" });
    s.addText(k, {
      x: M + 0.4, y: y + 0.18, w: 5, h: 0.3, fontFace: BFONT, fontSize: 10,
      bold: true, color: C.cyan, charSpacing: 1.5, margin: 0,
    });
    s.addText(v, {
      x: M + 0.4, y: y + 0.5, w: W - M * 2 - 0.8, h: 0.36, fontFace: "Courier New",
      fontSize: 14, color: C.paper, margin: 0,
    });
    s.addText(note, {
      x: M + 0.4, y: y + 0.86, w: W - M * 2 - 0.8, h: 0.3, fontFace: BFONT,
      fontSize: 10.5, color: "8395A2", margin: 0,
    });
  });

  s.addText("What a reviewer runs, with no manual edits:", {
    x: M, y: 5.75, w: 6, h: 0.3, fontFace: BFONT, fontSize: 11, color: "8395A2", margin: 0,
  });
  s.addText("pip install -r requirements-inference.txt\npython localize.py reference.png search.png", {
    x: M, y: 6.08, w: 9.5, h: 0.75, fontFace: "Courier New", fontSize: 13, color: C.cyan, margin: 0,
  });
  s.addNotes("Replace both «...» placeholders with the real URLs before submitting.");
}

// ============================================================== SLIDE 9 ====
{
  const s = lightSlide("References",
    "Per-choice justification in CITATIONS.md — 15 sections covering every augmentation and structural parameter");

  const groups = [
    ["SEM image formation", [
      "Goldstein, J. et al. Scanning Electron Microscopy and X-Ray Microanalysis (edge effect; charging; detector collection efficiency).",
      "Reimer, L. (1998). Scanning Electron Microscopy: Physics of Image Formation and Microanalysis, 2nd ed., Springer.",
      "Cazaux, J. (2004). Charging in scanning electron microscopy from inside and outside the specimen.",
      "Erasmus, S. J. & Smith, K. C. A. (1982). An automatic focusing and astigmatism correction system for the SEM.",
    ]],
    ["Noise modelling", [
      "Zhang, Y. et al. (2019). A Poisson-Gaussian Denoising Dataset with Real Fluorescence Microscopy Images, CVPR.",
      "Foi, A. et al. (2008). Practical Poissonian-Gaussian noise modeling for raw-data images.",
      "Healey, G. E. & Kondepudy, R. (1994). Radiometric CCD camera calibration and noise estimation.",
      "Janesick, J. R. (2001). Scientific Charge-Coupled Devices, SPIE Press.",
    ]],
    ["Device structure & pitch", [
      "imec, DRAM peripheral transistors technology platform.",
      "SemiAnalysis, The Memory Wall: Past, Present and Future of DRAM (pitch / feature-size / cell-area).",
      "ASIC North, FinFET Technology and Layout, Part 1 (fin pitch, contacted poly pitch).",
      "Sicard, E. Introducing 5-nm FinFET technology (fin and gate pitches by node).",
      "Bunday, B. D. et al., fin/gate metrology dimensions (arXiv:1503.06617, Table 10).",
    ]],
    ["Localization & the ambiguity problem", [
      "Lewis, J. P. (1995). Fast Normalized Cross-Correlation, Vision Interface.",
      "US Patent 9,430,457 — Ambiguity reduction for image alignment applications.",
      "US Patent 11,481,922 — Online navigational drift correction for metrology measurements.",
      "Reddy, B. S. & Chatterji, B. N. (1996). An FFT-based technique for translation, rotation and scale-invariant registration.",
      "Foroosh, H. et al. (2002). Extension of phase correlation to subpixel registration.",
      "Mack, C. (2007). Fundamental Principles of Optical Lithography; Levinson, H. J. (2010). Principles of Lithography.",
    ]],
  ];
  groups.forEach(([title, refs], i) => {
    const x = M + (i % 2) * 6.15, y = 1.78 + Math.floor(i / 2) * 2.42;
    card(s, { x, y, w: 5.85, h: 2.22 });
    s.addText(title, {
      x: x + 0.28, y: y + 0.16, w: 5.3, h: 0.3, fontFace: BFONT, fontSize: 11,
      bold: true, color: C.cyan, charSpacing: 1, margin: 0,
    });
    bullets(s, refs, { x: x + 0.28, y: y + 0.5, w: 5.3, h: 1.6, size: 8.8, color: C.muted });
  });
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote " + OUT));
