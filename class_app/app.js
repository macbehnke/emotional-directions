const data = window.EMO_DATA;

const labels = {
  en: "English",
  pl: "Polish",
  zh: "Chinese",
  gemini: "Gemini",
  qwen3_embedding_8b: "Qwen3-Embedding-8B",
  arctic_embed_l_v2: "Arctic Embed L v2",
  bielik_1_5b_v3: "Bielik",
  mmlw_roberta_large: "MMLW-RoBERTa",
};

const controls = {
  model: document.querySelector("#modelSelect"),
  language: document.querySelector("#languageSelect"),
  category: document.querySelector("#categorySelect"),
  emotion: document.querySelector("#emotionSelect"),
};

let activeTab = "languages";

const emotionOrder = [
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

function label(value) {
  return labels[value] || value;
}

function fillSelect(select, values, preferred) {
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label(value);
    select.appendChild(option);
  });
  if (values.includes(preferred)) select.value = preferred;
}

function formatNumber(value) {
  return Number(value).toFixed(3);
}

function renderRows() {
  const model = controls.model.value;
  const language = controls.language.value;
  const category = controls.category.value;
  const emotion = controls.emotion.value;
  const rows = data.examples
    .filter((row) => (
      row.model === model
      && row.language === language
      && row.category === category
      && row.emotion === emotion
    ))
    .sort((a, b) => a.rank - b.rank);

  const terms = data.term_map[emotion]?.[language];
  document.querySelector("#centroidText").textContent = terms
    ? `Emotion centroid: ${terms.emotion_terms}. Neutral centroid: ${terms.neutral_terms}.`
    : "No centroid terms available for this selection.";

  const equation = document.querySelector("#miniEquation");
  equation.innerHTML = `
    <span>${category}</span>
    <strong>+</strong>
    <span>${emotion}</span>
    <strong>-</strong>
    <span>neutral</span>
    <strong>=</strong>
    <span>${rows.slice(0, 3).map((row) => row.candidate).join(" / ") || "no candidates"}</span>
  `;
  renderComparison();
}

function candidateRows({ model, language, category, emotion }) {
  return data.examples
    .filter((row) => (
      row.model === model
      && row.language === language
      && row.category === category
      && row.emotion === emotion
    ))
    .sort((a, b) => a.rank - b.rank);
}

function formatCandidate(row) {
  if (!row) return "";
  return `
    <strong>${row.candidate}</strong>
    <span class="score-note">cos ${formatNumber(row.cosine_similarity)} · proj ${formatNumber(row.projection_on_emotion_direction)}</span>
  `;
}

function renderComparison() {
  const container = document.querySelector("#comparisonTable");
  const robust = document.querySelector("#robustConcepts");
  const model = controls.model.value;
  const language = controls.language.value;
  const category = controls.category.value;
  const emotion = controls.emotion.value;

  if (activeTab === "languages") {
    robust.classList.remove("visible");
    robust.innerHTML = "";
    const languages = ["en", "pl"];
    const rowsByLanguage = Object.fromEntries(languages.map((lang) => [
      lang,
      candidateRows({ model, language: lang, category, emotion }),
    ]));
    container.innerHTML = `
      <div class="comparison-grid languages">
        <div class="comparison-cell comparison-head">Rank</div>
        ${languages.map((lang) => `<div class="comparison-cell comparison-head">${label(model)} · ${label(lang)}</div>`).join("")}
        ${Array.from({ length: 10 }, (_, index) => `
          <div class="comparison-cell rank-cell">${index + 1}</div>
          ${languages.map((lang) => `<div class="comparison-cell">${formatCandidate(rowsByLanguage[lang][index])}</div>`).join("")}
        `).join("")}
      </div>
    `;
    return;
  }

  const models = ["gemini", "qwen3_embedding_8b", "arctic_embed_l_v2", "bielik_1_5b_v3", "mmlw_roberta_large"];
  const rowsByModel = Object.fromEntries(models.map((modelName) => [
    modelName,
    candidateRows({ model: modelName, language, category, emotion }),
  ]));
  const counts = new Map();
  models.forEach((modelName) => {
    rowsByModel[modelName].forEach((row) => {
      const key = row.candidate.trim().toLowerCase();
      const entry = counts.get(key) || { candidate: row.candidate, models: new Set(), bestRank: row.rank };
      entry.models.add(modelName);
      entry.bestRank = Math.min(entry.bestRank, row.rank);
      counts.set(key, entry);
    });
  });
  const shared = [...counts.values()]
    .filter((entry) => entry.models.size >= 2)
    .sort((a, b) => b.models.size - a.models.size || a.bestRank - b.bestRank)
    .slice(0, 8);
  if (shared.length) {
    robust.classList.add("visible");
    robust.innerHTML = `<strong>Robust across models:</strong> ${shared
      .map((entry) => `${entry.candidate} (${entry.models.size} models)`)
      .join(", ")}`;
  } else {
    robust.classList.add("visible");
    robust.innerHTML = "<strong>Robust across models:</strong> no exact overlap in the top-10 lists.";
  }
  container.innerHTML = `
    <div class="comparison-grid models">
      <div class="comparison-cell comparison-head">Rank</div>
      ${models.map((modelName) => `<div class="comparison-cell comparison-head">${label(modelName)} · ${label(language)}</div>`).join("")}
      ${Array.from({ length: 10 }, (_, index) => `
        <div class="comparison-cell rank-cell">${index + 1}</div>
        ${models.map((modelName) => `<div class="comparison-cell">${formatCandidate(rowsByModel[modelName][index])}</div>`).join("")}
      `).join("")}
    </div>
  `;
}

function init() {
  fillSelect(controls.model, data.meta.models, "gemini");
  fillSelect(controls.language, data.meta.languages, "en");
  fillSelect(controls.category, data.meta.categories, "food");
  fillSelect(
    controls.emotion,
    emotionOrder.filter((emotion) => data.meta.emotions.includes(emotion)),
    "positive",
  );
  Object.values(controls).forEach((control) => {
    control.addEventListener("change", renderRows);
  });
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      activeTab = button.dataset.tab;
      document.querySelectorAll(".tab").forEach((tab) => {
        tab.classList.toggle("active", tab === button);
      });
      renderComparison();
    });
  });
  renderRows();
}

init();
