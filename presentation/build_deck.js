const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const data = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const outPath = path.join(__dirname, "emotional_directions_class_presentation.pptx");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "Emotional Directions Project";
pptx.subject = "Class presentation";
pptx.title = "Emotional directions in embedding spaces";
pptx.company = "AMU";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "en-US",
};

const W = 13.333;
const H = 7.5;
const C = {
  ink: "16201C",
  muted: "5F706A",
  pale: "F5F8F4",
  line: "D9E4DD",
  green: "0F7C66",
  blue: "2F68B8",
  gold: "B87516",
  rose: "B84257",
  white: "FFFFFF",
  dark: "10231E",
};

function addTitle(slide, title, subtitle) {
  slide.addText(title, {
    x: 0.55, y: 0.38, w: 8.9, h: 0.55,
    fontFace: "Aptos Display", fontSize: 27, bold: true, color: C.ink,
    margin: 0,
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 0.57, y: 0.95, w: 9.4, h: 0.34,
      fontSize: 11.5, color: C.muted, margin: 0,
    });
  }
  slide.addShape(pptx.ShapeType.line, {
    x: 0.55, y: 1.36, w: 1.2, h: 0,
    line: { color: C.green, width: 2.2 },
  });
}

function addFooter(slide, text = "Emotional directions project · final run, 10 emotion labels") {
  slide.addText(text, {
    x: 0.55, y: 7.05, w: 7.8, h: 0.18,
    fontSize: 7.5, color: "80918A", margin: 0,
  });
}

function pill(slide, text, x, y, w, color = C.green) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h: 0.34,
    rectRadius: 0.08,
    fill: { color: "FFFFFF", transparency: 0 },
    line: { color },
  });
  slide.addText(text, {
    x: x + 0.08, y: y + 0.075, w: w - 0.16, h: 0.16,
    fontSize: 8.5, bold: true, color, align: "center", margin: 0,
  });
}

function bar(slide, label, value, max, x, y, w, color, suffix = "") {
  slide.addText(label, { x, y: y - 0.02, w: 2.05, h: 0.18, fontSize: 9, color: C.ink, margin: 0 });
  slide.addShape(pptx.ShapeType.rect, { x: x + 2.15, y, w, h: 0.12, fill: { color: "E8F0EB" }, line: { color: "E8F0EB" } });
  slide.addShape(pptx.ShapeType.rect, { x: x + 2.15, y, w: Math.max(0.05, w * value / max), h: 0.12, fill: { color }, line: { color } });
  slide.addText(`${value.toFixed(3)}${suffix}`, { x: x + 2.25 + w, y: y - 0.045, w: 0.62, h: 0.18, fontSize: 8, color: C.muted, margin: 0, align: "right" });
}

function cover() {
  const slide = pptx.addSlide();
  slide.background = { color: C.dark };
  slide.addText("Emotional directions", {
    x: 0.65, y: 0.68, w: 7.8, h: 0.58,
    fontFace: "Aptos Display", fontSize: 34, bold: true, color: "FFFFFF",
    margin: 0,
  });
  slide.addText("in embedding spaces", {
    x: 0.65, y: 1.26, w: 6.5, h: 0.42,
    fontFace: "Aptos Display", fontSize: 24, color: "DCEDE7",
    margin: 0,
  });
  slide.addText("A classroom walkthrough of vector arithmetic, emotion centroids, and model comparison", {
    x: 0.67, y: 6.25, w: 7.7, h: 0.25,
    fontSize: 12.5, color: "BBD0C9", margin: 0,
  });
  const tokens = [
    ["food", 8.6, 1.0], ["+", 9.45, 1.0], ["disgust", 9.85, 1.0], ["-", 10.95, 1.0],
    ["neutral", 11.35, 1.0], ["→", 9.45, 1.75], ["rotten food", 9.85, 1.75],
  ];
  tokens.forEach(([t, x, y], i) => {
    if (["+", "-", "→"].includes(t)) {
      slide.addText(t, { x, y, w: 0.35, h: 0.25, fontSize: 18, bold: true, color: "BBD0C9", margin: 0, align: "center" });
    } else {
      pill(slide, t, x, y, i === 6 ? 1.6 : 0.95, i === 6 ? "D99A2B" : "61B39E");
    }
  });
  slide.addShape(pptx.ShapeType.arc, { x: 8.1, y: 3.0, w: 3.8, h: 2.0, line: { color: "61B39E", width: 2, transparency: 20 }, adjustPoint: 0.3 });
  slide.addShape(pptx.ShapeType.line, { x: 9.2, y: 4.05, w: 2.1, h: -0.75, line: { color: "D99A2B", width: 3, beginArrowType: "none", endArrowType: "triangle" } });
  slide.addText("category + emotion − neutral", { x: 8.25, y: 5.55, w: 4.0, h: 0.25, fontSize: 13, color: "DCEDE7", margin: 0, align: "center" });
}

function slideIdea() {
  const slide = pptx.addSlide();
  slide.background = { color: C.pale };
  addTitle(slide, "The basic idea: words and phrases as positions", "Embedding models turn text into vectors. Similar meanings should live near each other.");
  const items = [
    ["food", 1.15, 2.05, C.green],
    ["restaurant meal", 2.2, 2.8, C.green],
    ["rotten food", 3.3, 2.25, C.rose],
    ["hospital food", 4.1, 3.05, C.rose],
    ["friend", 8.1, 2.2, C.blue],
    ["beloved grandmother", 9.0, 2.8, C.blue],
    ["dangerous street", 7.8, 4.7, C.gold],
    ["cemetery", 9.6, 4.25, C.gold],
  ];
  items.forEach(([t, x, y, color]) => pill(slide, t, x, y, Math.max(0.85, String(t).length * 0.09), color));
  slide.addShape(pptx.ShapeType.line, { x: 1.0, y: 5.85, w: 10.9, h: 0, line: { color: C.line, width: 1.2, endArrowType: "triangle" } });
  slide.addShape(pptx.ShapeType.line, { x: 1.0, y: 5.85, w: 0, h: -4.5, line: { color: C.line, width: 1.2, endArrowType: "triangle" } });
  slide.addText("A vector space is not a map of the mind, but it is a useful geometry of language.", { x: 0.75, y: 6.35, w: 10.7, h: 0.28, fontSize: 15, bold: true, color: C.ink, margin: 0 });
  addFooter(slide);
}

function slideCentroids() {
  const slide = pptx.addSlide();
  addTitle(slide, "Why centroids?", "Single words are brittle; averaging related seeds gives a more stable target.");
  slide.addText("Emotion centroid", { x: 0.85, y: 1.85, w: 2.4, h: 0.25, fontSize: 18, bold: true, color: C.green, margin: 0 });
  ["feeling disgusted", "feeling revolted", "feeling repulsed"].forEach((t, i) => pill(slide, t, 0.85, 2.35 + i * 0.48, 1.75, C.green));
  slide.addText("mean vector", { x: 3.05, y: 2.75, w: 1.2, h: 0.25, fontSize: 13, bold: true, color: C.ink, margin: 0 });
  slide.addShape(pptx.ShapeType.line, { x: 2.75, y: 2.85, w: 1.95, h: 0, line: { color: C.green, width: 2.4, endArrowType: "triangle" } });
  pill(slide, "disgust centroid", 4.9, 2.65, 1.55, C.green);
  slide.addText("Neutral centroid", { x: 7.05, y: 1.85, w: 2.4, h: 0.25, fontSize: 18, bold: true, color: C.gold, margin: 0 });
  ["neutral", "ordinary", "plain", "average"].forEach((t, i) => pill(slide, t, 7.05, 2.25 + i * 0.42, 1.25, C.gold));
  slide.addShape(pptx.ShapeType.line, { x: 8.6, y: 2.85, w: 1.95, h: 0, line: { color: C.gold, width: 2.4, endArrowType: "triangle" } });
  pill(slide, "neutral centroid", 10.75, 2.65, 1.55, C.gold);
  slide.addText("Final operation", { x: 0.85, y: 5.1, w: 2.0, h: 0.25, fontSize: 16, bold: true, color: C.ink, margin: 0 });
  slide.addText("embedding(category) + mean(emotion phrases) − mean(neutral terms)", { x: 2.2, y: 5.08, w: 8.8, h: 0.3, fontSize: 19, color: C.ink, bold: true, margin: 0 });
  addFooter(slide);
}

function slideDesign() {
  const slide = pptx.addSlide();
  slide.background = { color: C.pale };
  addTitle(slide, "Experiment design", "Same categories, emotions, languages, and controls across models.");
  const cols = [
    ["Models", ["Gemini", "Qwen3-Embedding-8B", "Bielik hidden states", "MMLW-RoBERTa"]],
    ["Languages", ["English", "Polish", "Chinese"]],
    ["Categories", ["food", "person", "object", "place", "situation"]],
    ["Emotion labels", ["10 labels", "discrete + broad affective", "centroid phrases"]],
  ];
  cols.forEach(([title, lines], i) => {
    const x = 0.75 + i * 3.05;
    slide.addText(title, { x, y: 1.95, w: 2.4, h: 0.28, fontSize: 16, bold: true, color: C.green, margin: 0 });
    lines.forEach((line, j) => slide.addText(line, { x, y: 2.45 + j * 0.43, w: 2.5, h: 0.22, fontSize: 11.5, color: C.ink, margin: 0 }));
  });
  slide.addShape(pptx.ShapeType.line, { x: 0.82, y: 4.8, w: 11.7, h: 0, line: { color: C.line, width: 1 } });
  slide.addText("Controls", { x: 0.85, y: 5.28, w: 1.5, h: 0.25, fontSize: 15, bold: true, color: C.ink, margin: 0 });
  ["emotion", "identity", "random", "shuffled emotion"].forEach((t, i) => pill(slide, t, 2.05 + i * 2.15, 5.2, 1.55, [C.green, C.blue, C.gold, C.rose][i]));
  slide.addText("The shuffled-emotion control is the hardest check: it asks whether the model separates the target emotion from another emotion, not just emotion from neutral.", { x: 0.85, y: 6.05, w: 10.9, h: 0.34, fontSize: 12.5, color: C.muted, margin: 0 });
  addFooter(slide);
}

function slideExamples() {
  const slide = pptx.addSlide();
  addTitle(slide, "Examples: the operation often gives intuitive neighbors", "Top candidates for Gemini and Qwen in English and Polish.");
  const examples = [
    ["food + disgust", "rotten food · spoiled milk · zepsute jedzenie", C.rose],
    ["person + love", "friend · beloved grandmother · kochana babcia", C.green],
    ["place + fear", "dangerous street · cemetery · niebezpieczna ulica", C.gold],
    ["situation + excitement", "hearing good news · celebration · świętowanie", C.blue],
  ];
  examples.forEach(([head, body, color], i) => {
    const y = 1.75 + i * 1.1;
    slide.addText(head, { x: 0.9, y, w: 2.5, h: 0.25, fontSize: 15, bold: true, color, margin: 0 });
    slide.addText(body, { x: 3.05, y, w: 8.5, h: 0.25, fontSize: 15, color: C.ink, margin: 0 });
    slide.addShape(pptx.ShapeType.line, { x: 0.9, y: y + 0.45, w: 10.8, h: 0, line: { color: C.line, width: 0.7 } });
  });
  slide.addText("These are nearest neighbors from category-specific candidate lists, so category retention is controlled by design.", { x: 0.9, y: 6.35, w: 10.8, h: 0.25, fontSize: 11.5, color: C.muted, margin: 0 });
  addFooter(slide);
}

function slideStrategy() {
  const slide = pptx.addSlide();
  addTitle(slide, "Strategy choice: phrase centroids win", "The supplementary run compared centroid phrases against single-word emotion seeds.");
  const rows = data.strategy.sort((a, b) => b.final_rank_score - a.final_rank_score);
  const max = Math.max(...rows.map(r => r.final_rank_score));
  rows.forEach((r, i) => bar(slide, r.emotion_strategy, r.final_rank_score, max, 1.0, 2.1 + i * 0.75, 6.5, i === 0 ? C.green : C.gold));
  slide.addText("Interpretation", { x: 1.0, y: 4.35, w: 1.4, h: 0.25, fontSize: 16, bold: true, color: C.ink, margin: 0 });
  slide.addText("Centroid phrases combine context with seed stability. Single words remain useful as a robustness check for discrete-emotion specificity.", { x: 2.35, y: 4.33, w: 8.4, h: 0.42, fontSize: 13.5, color: C.muted, margin: 0, fit: "shrink" });
  addFooter(slide);
}

function slideModels() {
  const slide = pptx.addSlide();
  slide.background = { color: C.pale };
  addTitle(slide, "Model comparison", "Final score combines projection and control deltas.");
  const rows = data.model.sort((a, b) => b.final_rank_score - a.final_rank_score);
  const max = Math.max(...rows.map(r => r.final_rank_score));
  const colors = { gemini: C.blue, qwen3_embedding_8b: C.green, bielik_1_5b_v3: C.gold, mmlw_roberta_large: C.rose };
  rows.forEach((r, i) => bar(slide, r.model.replace("qwen3_embedding_8b", "Qwen3").replace("bielik_1_5b_v3", "Bielik").replace("mmlw_roberta_large", "MMLW"), r.final_rank_score, max, 0.9, 1.9 + i * 0.58, 6.2, colors[r.model] || C.green));
  slide.addText("Takeaway", { x: 0.9, y: 4.7, w: 1.4, h: 0.25, fontSize: 16, bold: true, color: C.ink, margin: 0 });
  slide.addText("Gemini and Qwen are the cleanest embedding-model comparison. Bielik is interesting as an exploratory hidden-state condition; MMLW-RoBERTa should not support cross-lingual claims here.", { x: 2.0, y: 4.68, w: 9.5, h: 0.48, fontSize: 13.5, color: C.muted, margin: 0, fit: "shrink" });
  addFooter(slide);
}

function slideEmotionSpecificity() {
  const slide = pptx.addSlide();
  addTitle(slide, "Which emotions separated best?", "Delta against the shuffled-emotion control is the most direct specificity check.");
  const order = [
    "positive",
    "negative",
    "amusement",
    "anger",
    "disgust",
    "excitement",
    "fear",
    "joy",
    "love",
    "sadness",
  ];
  const rows = order
    .map((emotion) => data.emotion.find((row) => row.emotion === emotion))
    .filter(Boolean);
  const max = Math.max(...rows.map(r => r.delta_shuffled));
  rows.forEach((r, i) => bar(slide, r.emotion, r.delta_shuffled, max, i < 5 ? 0.85 : 6.95, 1.75 + (i % 5) * 0.6, 3.35, i < 3 ? C.green : C.blue));
  slide.addText("Positive and negative are broad affective poles. The remaining labels are discrete emotions shown alphabetically.", { x: 0.85, y: 5.65, w: 10.8, h: 0.42, fontSize: 14, color: C.muted, margin: 0 });
  addFooter(slide);
}

function slideApp() {
  const slide = pptx.addSlide();
  slide.background = { color: C.dark };
  slide.addText("Live demo", { x: 0.65, y: 0.55, w: 3.5, h: 0.4, fontFace: "Aptos Display", fontSize: 29, bold: true, color: C.white, margin: 0 });
  slide.addText("Use the browser app to switch model, language, category, and emotion, then compare outputs side by side.", { x: 0.67, y: 1.08, w: 8.8, h: 0.25, fontSize: 13.5, color: "C8DDD6", margin: 0 });
  ["Model", "Language", "Category", "Emotion"].forEach((t, i) => pill(slide, t, 0.85 + i * 2.15, 2.15, 1.45, "61B39E"));
  slide.addText("Ask the class:", { x: 0.85, y: 3.35, w: 2.0, h: 0.28, fontSize: 18, bold: true, color: "FFFFFF", margin: 0 });
  const prompts = [
    "Which outputs look interpretable?",
    "Where does Polish behave differently from English?",
    "Which model confuses broad valence with discrete emotion?",
  ];
  prompts.forEach((p, i) => slide.addText(p, { x: 1.05, y: 3.88 + i * 0.48, w: 8.7, h: 0.24, fontSize: 14, color: "DCEDE7", margin: 0 }));
  slide.addText("class_app/index.html", { x: 0.85, y: 6.38, w: 3.1, h: 0.25, fontSize: 13, bold: true, color: "F1C47B", margin: 0 });
}

function slideCaveats() {
  const slide = pptx.addSlide();
  addTitle(slide, "What we can and cannot claim", "This is a geometry-of-language experiment, not a direct map of human emotion.");
  const points = [
    ["Category retention", "Search was constrained to category-specific candidate lists."],
    ["Positive / negative", "These are broad affective poles, not discrete emotions."],
    ["Bielik", "Useful Polish LLM condition, but hidden-state pooling is experimental."],
    ["Next step", "Manual ratings: category fit, emotion fit, interpretability."],
  ];
  points.forEach(([head, body], i) => {
    const y = 1.75 + i * 0.95;
    slide.addText(head, { x: 0.9, y, w: 2.15, h: 0.25, fontSize: 15, bold: true, color: C.green, margin: 0 });
    slide.addText(body, { x: 3.0, y, w: 8.6, h: 0.25, fontSize: 14, color: C.ink, margin: 0 });
  });
  addFooter(slide);
}

cover();
slideIdea();
slideCentroids();
slideDesign();
slideExamples();
slideStrategy();
slideModels();
slideEmotionSpecificity();
slideApp();
slideCaveats();

pptx.writeFile({ fileName: outPath });
console.log(outPath);
