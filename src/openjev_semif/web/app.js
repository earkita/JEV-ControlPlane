(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const LETTERS = "ABCDEFGHIJKLMNOP";
  const timingLabels = {
    tokenization_s: "Tokenizacja", prefill_s: "Prefill", shared_prefill_s: "Wspólny prefill",
    branch_s: "Gałąź KV", scoring_s: "Scoring", total_s: "Łącznie",
  };
  const ui = {
    form: $("#decision-form"), state: $("#state-input"), stateJson: $("#state-json"),
    singleQuestion: $("#single-question"), singleOptions: $("#single-options"),
    questions: $("#question-list"), scorer: $("#scorer-input"),
    temperature: $("#temperature-input"), mode: $("#mode-input"),
    apiKey: $("#api-key"), result: $("#result-content"),
    empty: $("#result-empty"), error: $("#error-box"),
    badge: $("#output-badge"), actions: $("#result-actions"),
    run: $("#run-button"), runLabel: $("#run-label"),
  };
  let activeTab = "single";
  let lastResult = null;

  function element(tag, className = "", value = "") {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (value !== "") item.textContent = String(value);
    return item;
  }

  function optionRows(list) { return $$(".option-row", list); }

  function updateOptions(list) {
    const rows = optionRows(list);
    rows.forEach((row, index) => {
      $(".option-letter", row).textContent = list.closest(".question-card.score-type") ? String(index) : LETTERS[index];
      $(".remove-option", row).disabled = rows.length <= 2;
      $(".remove-option", row).setAttribute("aria-label", `Usuń opcję ${LETTERS[index]}`);
    });
    const add = list.parentElement.querySelector(".add-question-option") || $("#add-single-option");
    if (add) add.disabled = rows.length >= 16;
  }

  function addOption(list, data = {}) {
    if (optionRows(list).length >= 16) return;
    const row = $("#option-template").content.firstElementChild.cloneNode(true);
    $(".option-id", row).value = data.id || `option_${optionRows(list).length + 1}`;
    $(".option-description", row).value = data.description || "";
    $(".remove-option", row).addEventListener("click", () => {
      row.remove();
      updateOptions(list);
    });
    list.append(row);
    updateOptions(list);
    return row;
  }

  function questionCards() { return $$(".question-card", ui.questions); }

  function updateQuestionCards() {
    questionCards().forEach((card, index) => {
      $(".question-number", card).textContent = `QUESTION ${String(index + 1).padStart(2, "0")}`;
      $(".remove-question", card).disabled = questionCards().length <= 1;
    });
  }

  function setQuestionType(card) {
    const type = $(".question-type", card).value;
    $(".question-options", card).hidden = type === "noul";
    $(".noul-criteria", card).hidden = type !== "noul";
    card.classList.toggle("score-type", type === "score");
    $$(".option-letter", card).forEach((letter, index) => {
      letter.textContent = type === "score" ? String(index) : LETTERS[index];
    });
  }

  function addQuestion(data = {}) {
    const card = $("#question-template").content.firstElementChild.cloneNode(true);
    $(".question-id", card).value = data.id || `question_${questionCards().length + 1}`;
    $(".question-type", card).value = data.type || "choice";
    $(".question-instructions", card).value = data.instructions || "";
    $(".noul-true", card).value = data.trueText || "";
    $(".noul-false", card).value = data.falseText || "";
    const options = $(".option-list", card);
    (data.options || [{ id: "yes", description: "Tak" }, { id: "no", description: "Nie" }])
      .forEach((option) => addOption(options, option));
    $(".add-question-option", card).addEventListener("click", () => {
      addOption(options);
      setQuestionType(card);
    });
    $(".question-type", card).addEventListener("change", () => setQuestionType(card));
    $(".remove-question", card).addEventListener("click", () => {
      card.remove();
      updateQuestionCards();
    });
    ui.questions.append(card);
    setQuestionType(card);
    updateQuestionCards();
    return card;
  }

  function clearResult() {
    lastResult = null;
    ui.result.replaceChildren();
    ui.result.hidden = true;
    ui.empty.hidden = false;
    ui.error.hidden = true;
    ui.actions.hidden = true;
    ui.badge.textContent = "AWAITING INPUT";
    ui.badge.className = "output-badge";
    $("h3", ui.empty).textContent = "Decyzja zaczyna się tutaj.";
    $("p", ui.empty).textContent = "Uzupełnij stan, pytanie i opcje, a potem uruchom scoring. Zobaczysz rozkład, pewność i metryki wykonania.";
  }

  function setTab(tab) {
    if (tab !== activeTab) clearResult();
    activeTab = tab;
    for (const name of ["single", "batch"]) {
      const selected = name === tab;
      const button = $(`#tab-${name}`);
      button.classList.toggle("active", selected);
      button.setAttribute("aria-selected", String(selected));
      $(`#panel-${name}`).hidden = !selected;
    }
    $("#mode-field").hidden = tab !== "batch";
    ui.runLabel.textContent = tab === "single" ? "Oceń decyzję" : "Oceń pytania";
  }

  function updateScorer() {
    const semif = ui.scorer.value === "semif";
    $("#scorer-help").textContent = semif
      ? "SemIf porównuje logity liter A/B/C w ostatniej pozycji. Prawdopodobieństwa dotyczą tylko wyświetlonych opcji."
      : "Likelihood ocenia P(tekst opcji | kontekst). Wyniki surowe mają inne znaczenie niż logity SemIf.";
    $("#mode-input option[value='shared']").disabled = !semif;
    if (!semif && ui.mode.value === "shared") ui.mode.value = "direct";
  }

  function loadPreset() {
    ui.state.value = "Build failed after enabling a new MXFP4 kernel. The previous build passed, and the error points to the kernel launch.";
    ui.stateJson.checked = false;
    ui.singleQuestion.value = "What should the agent do next?";
    ui.singleOptions.replaceChildren();
    addOption(ui.singleOptions, { id: "debug", description: "Inspect the failing kernel and logs" });
    addOption(ui.singleOptions, { id: "rollback", description: "Disable the optimization" });
    addOption(ui.singleOptions, { id: "retest", description: "Run the tests again" });
    ui.questions.replaceChildren();
    addQuestion({
      id: "next_action", instructions: "What should the agent do next?",
      options: [
        { id: "debug", description: "Inspect the failing kernel and logs" },
        { id: "rollback", description: "Disable the optimization" },
        { id: "retest", description: "Run the tests again" },
      ],
    });
    addQuestion({
      id: "validation", instructions: "What would confirm the repair?",
      options: [
        { id: "rerun", description: "Run the failing tests again" },
        { id: "ignore", description: "Ignore the failure" },
      ],
    });
    ui.scorer.value = "semif";
    ui.temperature.value = "1.00";
    ui.mode.value = "direct";
    setTab("single");
    updateScorer();
    clearResult();
  }

  function readState() {
    const raw = ui.state.value.trim();
    if (!raw) throw new Error("Wpisz stan przed uruchomieniem scoringu.");
    if (!ui.stateJson.checked) return raw;
    let parsed;
    try { parsed = JSON.parse(raw); }
    catch { throw new Error("Stan oznaczony jako JSON nie jest poprawnym JSON-em."); }
    if (parsed === null || typeof parsed !== "object")
      throw new Error("Stan JSON musi być obiektem albo tablicą.");
    return parsed;
  }

  function readOptions(list, type) {
    const rows = optionRows(list);
    if (rows.length < 2 || rows.length > 16) throw new Error("Pytanie wymaga od 2 do 16 opcji.");
    const seen = new Set();
    return rows.map((row, index) => {
      const description = $(".option-description", row).value.trim();
      if (!description) throw new Error(`Uzupełnij opis opcji ${index + 1}.`);
      if (type === "score") return description;
      const id = $(".option-id", row).value.trim();
      if (!id) throw new Error(`Uzupełnij ID opcji ${index + 1}.`);
      if (seen.has(id)) throw new Error(`ID opcji „${id}” powtarza się.`);
      seen.add(id);
      return { id, description };
    });
  }

  function buildRequest() {
    const state = readState();
    const temperature = Number(ui.temperature.value);
    if (!Number.isFinite(temperature) || temperature <= 0)
      throw new Error("Temperatura musi być dodatnią liczbą.");
    const scorer = ui.scorer.value;
    if (activeTab === "single") {
      const question = ui.singleQuestion.value.trim();
      if (!question) throw new Error("Wpisz pytanie.");
      return { path: "/score", body: {
        state, question, options: readOptions(ui.singleOptions, "choice"), scorer, temperature,
      }};
    }
    const cards = questionCards();
    if (!cards.length) throw new Error("Dodaj przynajmniej jedno pytanie.");
    if (ui.mode.value === "shared" && cards.length < 2)
      throw new Error("Tryb shared wymaga co najmniej dwóch pytań.");
    const questions = Object.create(null);
    for (const card of cards) {
      const id = $(".question-id", card).value.trim();
      if (!id) throw new Error("Każde pytanie wymaga ID.");
      if (Object.hasOwn(questions, id)) throw new Error(`ID pytania „${id}” powtarza się.`);
      const type = $(".question-type", card).value;
      const instructions = $(".question-instructions", card).value.trim();
      if (!instructions) throw new Error(`Uzupełnij instrukcję pytania „${id}”.`);
      let criteria;
      if (type === "noul") {
        const trueText = $(".noul-true", card).value.trim();
        const falseText = $(".noul-false", card).value.trim();
        if (!trueText || !falseText) throw new Error(`Uzupełnij oba warunki pytania „${id}”.`);
        criteria = { true: trueText, false: falseText };
      } else if (type === "score") {
        criteria = readOptions($(".option-list", card), type);
      } else {
        criteria = Object.fromEntries(readOptions($(".option-list", card), type)
          .map((option) => [option.id, option.description]));
      }
      questions[id] = { type, instructions, criteria };
    }
    return { path: "/v1/systemone", body: { state, questions, scorer, mode: ui.mode.value, temperature }};
  }

  function formatPercent(value) {
    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  function formatTime(value) {
    if (!Number.isFinite(Number(value))) return "—";
    const milliseconds = Number(value) * 1000;
    return milliseconds >= 1000 ? `${(milliseconds / 1000).toFixed(2)} s` : `${milliseconds.toFixed(1)} ms`;
  }

  function addMetric(parent, title, value) {
    const tile = element("div", "metric-tile");
    tile.append(element("span", "", title), element("strong", "", value));
    parent.append(tile);
  }

  function addMeta(parent, label, value) {
    const row = element("div", "meta-row");
    row.append(element("span", "", label), element("strong", "", value ?? "—"));
    parent.append(row);
  }

  function renderProbabilities(parent, rows, best) {
    const list = element("div", "probability-list");
    for (const item of rows) {
      const row = element("div", `probability-row${item.id === best ? " selected" : ""}`);
      const head = element("div", "probability-row-head");
      head.append(element("strong", "", item.description || item.id),
        element("span", "", formatPercent(item.probability)));
      const track = element("div", "probability-track");
      const fill = element("span", "probability-fill");
      fill.style.width = `${Math.max(0, Math.min(100, Number(item.probability) * 100))}%`;
      track.append(fill);
      const foot = element("div", "probability-foot");
      foot.append(element("span", "", item.id),
        element("span", "", item.rawScore == null ? "" : `RAW ${Number(item.rawScore).toFixed(3)}`));
      row.append(head, track, foot);
      list.append(row);
    }
    parent.append(list);
  }

  function renderTimings(parent, timings) {
    const available = Object.entries(timingLabels).filter(([key]) => timings && Number.isFinite(timings[key]));
    if (!available.length) return;
    parent.append(element("div", "result-divider"), element("p", "metric-section-title", "TIMING / SERVER"));
    const grid = element("div", "metrics-grid");
    available.forEach(([key, label]) => addMetric(grid, label, formatTime(timings[key])));
    parent.append(grid);
  }

  function renderMetadata(parent, data) {
    parent.append(element("div", "result-divider"), element("p", "metric-section-title", "REPRODUCIBILITY"));
    const list = element("div", "meta-list");
    addMeta(list, "Scorer / tryb", `${data.scorer || "—"} / ${data.mode || "direct"}`);
    addMeta(list, "Model rev.", data.model_revision);
    addMeta(list, "Tokenizer rev.", data.tokenizer_revision);
    addMeta(list, "Prompt SHA-256", data.prompt_hash);
    addMeta(list, "Temp.", data.temperature);
    addMeta(list, "Truncation", data.truncated ? "tak" : "nie");
    parent.append(list);
  }

  function renderSingle(data) {
    const root = ui.result;
    const selected = data.options.find((option) => option.option === data.best) || data.options[data.best_index];
    const kicker = element("div", "result-kicker");
    kicker.append(element("span", "", "TOP CHOICE"), element("span", "", data.scorer.toUpperCase()));
    root.append(kicker, element("h3", "winner", selected?.description || data.best),
      element("p", "result-subtitle", `ID: ${data.best} · Pewność (koncentracja rozkładu): ${formatPercent(data.confidence)}`));
    renderProbabilities(root, data.options.map((option) => ({
      id: option.option, description: option.description, probability: option.probability,
      rawScore: option.raw_score,
    })), data.best);
    const overview = element("div", "metrics-grid");
    addMetric(overview, "Pewność", formatPercent(data.confidence));
    addMetric(overview, "Tokeny wejściowe", data.input_tokens);
    root.append(overview);
    renderTimings(root, data.timings);
    renderMetadata(root, data);
    root.append(element("p", "settings-note", "Pewność mierzy koncentrację rozkładu; bez profilu kalibracji nie oznacza szansy poprawnej decyzji."));
  }

  function batchDescriptions(question) {
    if (question.type === "score")
      return Object.fromEntries(question.criteria.map((description, index) => [String(index), String(description)]));
    if (question.type === "noul") return { true: question.criteria?.true || "Tak", false: question.criteria?.false || "Nie" };
    return question.criteria;
  }

  function renderBatch(data, payload) {
    const root = ui.result;
    const count = Object.keys(data.answers).length;
    const kicker = element("div", "result-kicker");
    kicker.append(element("span", "", "SYSTEMONE / RESULTS"), element("span", "", payload.mode.toUpperCase()));
    root.append(kicker, element("h3", "winner", `${count} ${count === 1 ? "decyzja" : "decyzje"}`),
      element("p", "result-subtitle", `Wspólny stan · ${data.model} · ${payload.scorer}`));
    const summary = element("div", "batch-summary");
    addMetric(summary, "Pytania", count);
    addMetric(summary, "Tokeny łącznie", data.usage.input_tokens);
    root.append(summary);
    for (const [id, answer] of Object.entries(data.answers)) {
      const question = payload.questions[id];
      const descriptions = batchDescriptions(question);
      const card = element("article", "answer-card");
      const cardTop = element("div", "result-kicker");
      cardTop.append(element("span", "", id), element("span", "", `${answer.type.toUpperCase()} · ${answer.scorer.toUpperCase()}`));
      let best;
      let title;
      if (answer.type === "choice") {
        best = answer.choice;
        title = descriptions[best] || best;
      } else if (answer.type === "score") {
        best = Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1])[0]?.[0];
        title = `Ocena ${Number(answer.score).toFixed(2)}`;
      } else {
        best = answer.noul >= 0.5 ? "true" : "false";
        title = best === "true" ? "Tak" : "Nie";
      }
      card.append(cardTop, element("h4", "winner", title),
        element("p", "result-subtitle", question.instructions));
      renderProbabilities(card, Object.entries(answer.probabilities).map(([key, probability]) => ({
        id: key, description: String(descriptions[key] || key), probability,
      })), best);
      const small = element("div", "meta-list");
      addMeta(small, "Pewność", formatPercent(answer.confidence));
      addMeta(small, "Czas łącznie", formatTime(answer.timings.total_s));
      if (answer.timings.prefill_s) addMeta(small, "Prefill", formatTime(answer.timings.prefill_s));
      if (answer.timings.shared_prefill_s) addMeta(small, "Wspólny prefill", formatTime(answer.timings.shared_prefill_s));
      if (answer.timings.branch_s) addMeta(small, "Gałąź KV", formatTime(answer.timings.branch_s));
      if (answer.timings.shared_prefix_tokens) addMeta(small, "Prefiks tokenów", answer.timings.shared_prefix_tokens);
      addMeta(small, "Tokeny", answer.input_tokens);
      addMeta(small, "Prompt hash", answer.prompt_hash);
      card.append(small);
      root.append(card);
    }
    root.append(element("p", "settings-note", `Rewizja modelu: ${data.model_revision}. Prawdopodobieństwa są warunkowe względem widocznych opcji.`));
  }

  function showError(message) {
    ui.error.textContent = message;
    ui.error.hidden = false;
    ui.badge.textContent = "ERROR";
    ui.badge.className = "output-badge error";
    ui.empty.hidden = true;
    ui.result.hidden = true;
    ui.actions.hidden = true;
  }

  function detailMessage(body, status) {
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail))
      return body.detail.map((entry) => entry.msg || JSON.stringify(entry)).join("\n");
    return `Serwer zwrócił HTTP ${status}.`;
  }

  async function runDecision(event) {
    event.preventDefault();
    let request;
    try { request = buildRequest(); }
    catch (error) { showError(error.message); return; }
    ui.error.hidden = true;
    ui.badge.textContent = "RUNNING";
    ui.badge.className = "output-badge";
    ui.run.disabled = true;
    ui.runLabel.textContent = "Obliczanie…";
    ui.result.hidden = true;
    ui.empty.hidden = false;
    $("h3", ui.empty).textContent = "Model liczy decyzję…";
    $("p", ui.empty).textContent = "Odczytujemy logity opcji i przygotowujemy rozkład.";
    const headers = { "Content-Type": "application/json" };
    const apiKey = ui.apiKey.value.trim();
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    try {
      const response = await fetch(request.path, {
        method: "POST", headers, body: JSON.stringify(request.body),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) throw new Error(detailMessage(body, response.status));
      lastResult = body;
      ui.result.replaceChildren();
      if (request.path === "/score") renderSingle(body);
      else renderBatch(body, request.body);
      ui.empty.hidden = true;
      ui.result.hidden = false;
      ui.actions.hidden = false;
      ui.badge.textContent = "COMPLETE";
      ui.badge.className = "output-badge complete";
      if (window.innerWidth < 850) ui.result.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      showError(error.message || "Nie udało się połączyć z serwerem.");
    } finally {
      ui.run.disabled = false;
      ui.runLabel.textContent = activeTab === "single" ? "Oceń decyzję" : "Oceń pytania";
    }
  }

  async function refreshHealth() {
    const connection = $("#connection");
    const status = $("#connection-text");
    try {
      const response = await fetch("/health", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const health = await response.json();
      connection.className = `connection ${health.status === "ready" ? "ready" : ""}`;
      status.textContent = health.status === "ready" ? "MODEL READY" : "Ładowanie modelu";
      $("#model-name").textContent = health.model;
      $("#model-details").textContent = `${health.backend} · ${health.device} · ${health.dtype}`;
      $("#model-scorer").textContent = `SCORER ${String(health.scorer).toUpperCase()}`;
    } catch {
      connection.className = "connection error";
      status.textContent = "SERWER NIEDOSTĘPNY";
      $("#model-name").textContent = "Brak połączenia";
      $("#model-details").textContent = "Sprawdź proces serwera";
    }
  }

  async function copyResult() {
    if (!lastResult) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2));
      const button = $("#copy-json");
      button.textContent = "Skopiowano ✓";
      setTimeout(() => { button.textContent = "Kopiuj JSON"; }, 1800);
    } catch { showError("Przeglądarka nie pozwoliła skopiować wyniku. Użyj opcji pobierania."); }
  }

  function downloadResult() {
    if (!lastResult) return;
    const file = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(file);
    const link = element("a");
    link.href = url;
    link.download = "openjev-semif-result.json";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  $("#tab-single").addEventListener("click", () => setTab("single"));
  $("#tab-batch").addEventListener("click", () => setTab("batch"));
  $("#load-preset").addEventListener("click", loadPreset);
  $("#add-single-option").addEventListener("click", () => addOption(ui.singleOptions));
  $("#add-question").addEventListener("click", () => addQuestion());
  $("#refresh-health").addEventListener("click", refreshHealth);
  $("#copy-json").addEventListener("click", copyResult);
  $("#download-json").addEventListener("click", downloadResult);
  ui.scorer.addEventListener("change", updateScorer);
  ui.form.addEventListener("submit", runDecision);
  loadPreset();
  refreshHealth();
})();
