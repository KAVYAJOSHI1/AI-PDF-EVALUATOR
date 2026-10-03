"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (n) => `${Math.round(n)}%`;
const tone = (s) => (s >= 70 ? "good" : s >= 40 ? "warn" : "bad");

const VIEWS = {
  upload: ["Document", "Upload study material to begin."],
  insights: ["NLP Insights", "Every stage of the pipeline, computed on your document."],
  ask: ["Ask the Document", "Natural-language search with highlighted evidence."],
  exam: ["Exam", "Answer questions grounded in your document and get explainable feedback."],
  results: ["Results", "Your scores, weak spots and a revision plan."],
};
const SAMPLE = `Natural Language Processing (NLP) is a branch of Artificial Intelligence that enables computers to understand, interpret and generate human language. Tokenization is the process of dividing text into smaller units called tokens. Stop words are common words that usually carry little meaning. Stemming removes prefixes or suffixes to obtain the root form of a word. Lemmatization converts words to their dictionary base form, considering grammar and vocabulary. Bag of Words (BoW) converts text into numerical vectors by counting the frequency of each word. TF-IDF assigns higher weights to words that are frequent in one document but rare across the corpus. Word2Vec is a neural network-based algorithm that learns word meanings from their surrounding words. Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique that preserves as much variance as possible.`;

const S = { doc: null, lab: null, qs: [], res: {}, idx: 0, diff: "Medium", view: "upload" };

/* ---------- plumbing ---------- */
async function api(path, body, label = "Working…", quiet = false) {
  $("#loaderText").textContent = label;
  if (!quiet) $("#loader").classList.remove("hidden");
  try {
    const isForm = body instanceof FormData;
    const r = await fetch(path, body ? { method: "POST", headers: isForm ? {} : { "Content-Type": "application/json" }, body: isForm ? body : JSON.stringify(body) } : {});
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed");
    return data;
  } finally {
    $("#loader").classList.add("hidden");
  }
}
function toast(msg, err = false) {
  const t = document.createElement("div");
  t.className = "toast" + (err ? " err" : "");
  t.textContent = msg;
  $("#toasts").append(t);
  setTimeout(() => t.remove(), 5000);
}
const guard = (fn) => async (...a) => { try { await fn(...a); } catch (e) { toast(e.message || String(e), true); } };

function show(view) {
  S.view = view;
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $("#viewTitle").textContent = VIEWS[view][0];
  $("#viewSub").textContent = VIEWS[view][1];
  window.scrollTo({ top: 0 });
  if (view === "insights") guard(renderInsights)();
  if (view === "results") renderResults();
}
$$(".nav-item").forEach((b) => b.addEventListener("click", () => !b.disabled && show(b.dataset.view)));
document.addEventListener("click", (e) => { const g = e.target.closest("[data-go]"); if (g) show(g.dataset.go); });

/* theme */
const root = document.documentElement;
try { root.dataset.theme = localStorage.getItem("theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); } catch { root.dataset.theme = "light"; }
$("#themeBtn").onclick = () => { root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark"; try { localStorage.setItem("theme", root.dataset.theme); } catch {} };

/* ---------- 1. document ---------- */
const drop = $("#drop"), fileInput = $("#file");
drop.onclick = () => fileInput.click();
drop.onkeydown = (e) => e.key === "Enter" && fileInput.click();
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => upload(e.dataTransfer.files[0]));
fileInput.onchange = () => upload(fileInput.files[0]);

const upload = guard(async (file) => {
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  setDoc(await api("/api/upload", fd, "Extracting and analysing PDF…"));
});
$("#pasteBtn").onclick = guard(async () => {
  const text = $("#pasteText").value.trim();
  if (!text) return toast("Paste some text first.", true);
  setDoc(await api("/api/analyze", { text, filename: "pasted-text" }, "Analysing text…"));
});
$("#sampleBtn").onclick = () => { $("#pasteText").value = SAMPLE; $("#pasteBtn").click(); };

function setDoc(d) {
  const pages = d.pages || [];
  S.doc = { ...d, text: pages.map((p) => p.text).join("\n\n"), page_count: d.page_count ?? pages.length };
  Object.assign(S, { lab: null, qs: [], res: {}, idx: 0 });
  $("#examBody").innerHTML = "";
  const defined = d.topics.filter((t) => t.evidence === "Defined in the document").length;
  $("#stats").innerHTML = [[S.doc.page_count, "Pages"], [d.word_count.toLocaleString(), "Words"], [d.chunks.length, "Passages"], [defined || d.topics.length, defined ? "Defined concepts" : "Topics"]]
    .map(([v, l]) => `<div class="stat"><b>${esc(v)}</b><span>${l}</span></div>`).join("");
  $("#summary").textContent = d.summary;
  $("#topics").innerHTML = d.topics.map((t) => `<span class="chip">${esc(t.topic)}${t.source_page ? `<small>p${t.source_page}</small>` : ""}</span>`).join("");
  $("#docInfo").classList.remove("hidden");
  const chip = $("#docChip");
  chip.classList.remove("hidden");
  chip.innerHTML = `<b>${esc(d.filename)}</b><span class="muted">${S.doc.page_count} pages · ${d.word_count.toLocaleString()} words</span>`;
  $$(".nav-item").forEach((b) => (b.disabled = b.dataset.view === "results"));
  $("#nav .nav-item.done")?.classList.remove("done");
  (d.warnings || []).forEach((w) => toast(w));
  toast("Document analysed.");
}

/* ---------- 2. insights ---------- */
const POS_VAR = { NOUN: "--noun", VERB: "--verb", ADJ: "--adj", ADV: "--adv", OTHER: "--other" };
const bars = (items, key, label) => {
  const max = Math.max(...items.map((i) => i[key]), 1e-9);
  return items.map((i) => `<div class="bar-row"><span class="lbl" title="${esc(i[label])}">${esc(i[label])}</span><div class="bar"><i style="width:${(i[key] / max) * 100}%"></i></div><span class="v">${i[key].toFixed(2)}</span></div>`).join("");
};
const section = (n, title, sub, body) => `<div class="card"><div class="section-title"><span class="badge">${n}</span><h3 style="margin:0">${title}</h3></div><p class="sub">${sub}</p>${body}</div>`;

async function renderInsights() {
  const body = $("#insightsBody");
  if (!S.lab) {
    S.lab = await api("/api/lab", { text: S.doc.text, chunks: S.doc.chunks }, "Running the NLP pipeline…");  // fast heuristic NER; BERT loads in background
  }
  const L = S.lab;
  body.innerHTML =
    section("Graph", "Concept map", "Central concepts, linked when they appear in the same passage. Hover a node to see its neighbours.", `<svg id="graph" viewBox="0 0 900 500"></svg>`) +
    `<div class="grid two">` +
    section("TF-IDF", "Statistical keywords", "Frequent in one place, rare elsewhere.", bars(L.tfidf.slice(0, 10), "score", "term")) +
    section("TextRank", "Graph-based keyphrases", "Central in the word co-occurrence network (PageRank).", bars(L.keyphrases.slice(0, 10), "score", "phrase")) +
    `</div>` +
    section("POS", "Part-of-speech tagging", "NLTK averaged perceptron tagger.", `<div class="pos">${L.pos_tags.map(([t, g]) => `<span title="${g}" style="color:var(${POS_VAR[g] || "--other"})">${esc(t)}</span>`).join(" ")}</div><div class="legend">${Object.entries(POS_VAR).map(([k, v]) => `<span><i style="background:var(${v})"></i>${k}</span>`).join("")}</div>`) +
    `<div class="grid two">` +
    section("Lemma", "Tokens → lemmas", "POS-aware WordNet lemmatisation.", `<table><tr><th>Token</th><th>Lemma</th></tr>${L.lemmas.slice(0, 12).map((r) => `<tr><td>${esc(r.token)}</td><td>${esc(r.lemma)}</td></tr>`).join("")}</table>`) +
    section("N-grams", "Frequent phrases", "Bigrams and trigrams of lemmas.", `<div class="chips">${[...L.bigrams, ...L.trigrams].slice(0, 16).map(([t, c]) => `<span class="chip">${esc(t)}<small>×${c}</small></span>`).join("")}</div>`) +
    `</div>` +
    section("Rank", "Most important sentences", "TextRank over a TF-IDF similarity graph: the six highest-ranked sentences, in document order.", L.sentences.map((s) => `<div class="sent top">${esc(s.sentence)}<small>score ${s.score.toFixed(3)}</small></div>`).join("")) +
    section("NER", "Named entities", `<label class="toggle"><input type="checkbox" id="nerToggle" checked> Local BERT model (uncheck for the fast heuristic)</label>`, `<div id="ents"></div>`);
  const loadEntities = async (useNer) => {
    $("#ents").innerHTML = `<p class="muted">${useNer ? "Running BERT NER…" : "Loading…"}</p>`;
    try { drawEntities((await api("/api/lab", { text: S.doc.text, chunks: S.doc.chunks, use_ner: useNer }, "", true)).entities); }
    catch (e) { $("#ents").innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
  };
  if (S.view === "insights") loadEntities(true);
  $("#nerToggle").onchange = (e) => loadEntities(e.target.checked);
  drawGraph(L.graph);
}
function drawEntities(ents) {
  $("#ents").innerHTML = ents.length ? ents.map((e) => `<span class="ent">${esc(e.text)}<em>${esc(e.label)}</em></span>`).join("") : `<p class="muted">No entities found.</p>`;
}

function drawGraph({ nodes, edges }) {
  const svg = $("#graph");
  if (!nodes.length) { svg.outerHTML = `<p class="muted">Not enough concepts to draw a map.</p>`; return; }
  const W = 900, H = 500, idx = Object.fromEntries(nodes.map((n, i) => [n.id, i]));
  nodes.forEach((n, i) => { const a = (i / nodes.length) * Math.PI * 2; n.x = W / 2 + Math.cos(a) * 280; n.y = H / 2 + Math.sin(a) * 190; n.vx = n.vy = 0; });
  for (let it = 0; it < 320; it++) {
    const cool = 1 - it / 320;
    nodes.forEach((a, i) => nodes.forEach((b, j) => {
      if (i >= j) return;
      let dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy + 0.01, f = (16000 / d2) * cool, d = Math.sqrt(d2);
      dx /= d; dy /= d; a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }));
    edges.forEach((e) => {
      const a = nodes[idx[e.source]], b = nodes[idx[e.target]], dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) || 1, f = (d - 170) * 0.01 * Math.min(e.weight, 3) * cool;
      a.vx += (dx / d) * f; a.vy += (dy / d) * f; b.vx -= (dx / d) * f; b.vy -= (dy / d) * f;
    });
    nodes.forEach((n) => {
      n.vx += (W / 2 - n.x) * 0.004; n.vy += (H / 2 - n.y) * 0.006;
      n.x = Math.max(70, Math.min(W - 70, n.x + n.vx * 0.85)); n.y = Math.max(34, Math.min(H - 34, n.y + n.vy * 0.85)); n.vx *= 0.6; n.vy *= 0.6;
    });
  }
  const maxW = Math.max(...nodes.map((n) => n.weight), 1);
  svg.innerHTML =
    edges.map((e) => { const a = nodes[idx[e.source]], b = nodes[idx[e.target]]; return `<line class="edge" data-s="${esc(e.source)}" data-t="${esc(e.target)}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke-width="${Math.min(1 + e.weight, 5)}"/>`; }).join("") +
    nodes.map((n) => `<g class="node" data-id="${esc(n.id)}"><circle cx="${n.x}" cy="${n.y}" r="${7 + (n.weight / maxW) * 9}"/><text x="${n.x}" y="${n.y - 14 - (n.weight / maxW) * 6}" text-anchor="middle">${esc(n.label)}</text></g>`).join("");
  $$(".node", svg).forEach((g) => {
    g.onmouseenter = () => {
      const id = g.dataset.id, near = new Set([id]);
      $$(".edge", svg).forEach((l) => { const on = l.dataset.s === id || l.dataset.t === id; l.classList.toggle("hl", on); if (on) { near.add(l.dataset.s); near.add(l.dataset.t); } });
      $$(".node", svg).forEach((n) => n.classList.toggle("dim", !near.has(n.dataset.id)));
    };
    g.onmouseleave = () => { $$(".edge", svg).forEach((l) => l.classList.remove("hl")); $$(".node", svg).forEach((n) => n.classList.remove("dim")); };
  });
}

/* ---------- 3. ask ---------- */
const markTerms = (sentence, terms) => esc(sentence).replace(/[A-Za-z][A-Za-z0-9'-]*/g, (w) => (terms.some((t) => t.length > 2 && w.toLowerCase().startsWith(t)) ? `<mark>${w}</mark>` : w));
$("#askForm").onsubmit = guard(async (e) => {
  e.preventDefault();
  const query = $("#askInput").value.trim();
  if (!query) return;
  const { hits } = await api("/api/search", { query, chunks: S.doc.chunks, top_k: 3 }, "Searching…");
  $("#askResults").innerHTML = hits.length ? hits.map((h, i) => `
    <div class="hit">
      <div class="meta"><span class="chip">#${i + 1}</span><span>Page ${h.page} · ${esc(h.section)}</span><span class="tag">BM25 ${h.bm25.toFixed(2)}</span><span class="tag">semantic ${h.semantic.toFixed(2)}</span>${h.rerank != null ? `<span class="tag diff">rerank ${h.rerank.toFixed(2)}</span>` : ""}</div>
      <div class="answer">${markTerms(h.answer_sentence, h.matched_terms)}</div>
      <details><summary>Full passage</summary><p>${esc(h.passage)}</p></details>
    </div>`).join("") : `<div class="empty">No relevant passage found.</div>`;
});

/* ---------- 4. exam ---------- */
$$("#qDiff button").forEach((b) => (b.onclick = () => { $$("#qDiff button").forEach((x) => x.classList.remove("on")); b.classList.add("on"); S.diff = b.dataset.v; }));
$("#genBtn").onclick = guard(async () => {
  const count = Math.max(1, Math.min(20, +$("#qCount").value || 5));
  const { questions } = await api("/api/questions", { text: S.doc.text, topics: S.doc.topics, chunks: S.doc.chunks, count, difficulty: S.diff }, "Generating questions…");
  if (!questions.length) return toast("This document doesn't contain enough material for questions at this level. Try Easy.", true);
  if (questions.length < count) toast(`Only ${questions.length} well-supported ${S.diff} questions can be asked from this document.`);
  Object.assign(S, { qs: questions, res: {}, idx: 0 });
  $$(".nav-item")[3].classList.remove("done");
  renderExam();
});

function renderExam() {
  const q = S.qs[S.idx], r = S.res[q.id];
  $("#examBody").innerHTML = `
    <div class="exam">
      <div class="qlist">${S.qs.map((x, i) => `<button class="qitem ${i === S.idx ? "active" : ""}" data-i="${i}"><span class="n">${i + 1}</span><span>Question ${i + 1}</span>${S.res[x.id] ? `<span class="s" style="color:var(--${tone(S.res[x.id].overall_score)})">${pct(S.res[x.id].overall_score)}</span>` : ""}</button>`).join("")}</div>
      <div class="card">
        <div class="qmeta"><span class="tag diff">${esc(q.difficulty)}</span><span class="tag">Page ${q.source_page}</span><span class="tag">Topic: ${esc(q.topic)}</span></div>
        <p class="qprompt">${esc(q.prompt)}</p>
        <textarea id="ans" class="answerbox" placeholder="Write your answer in your own words…">${esc(q._draft || "")}</textarea>
        <div class="row" style="justify-content:space-between;align-items:center"><span class="counter" id="wc">0 words</span>
          <div class="row" style="margin:0"><button class="btn ghost" id="prevQ" ${S.idx === 0 ? "disabled" : ""}>← Prev</button><button class="btn" id="submitQ">Evaluate answer</button><button class="btn ghost" id="nextQ" ${S.idx === S.qs.length - 1 ? "disabled" : ""}>Next →</button></div></div>
        <div id="resultBox">${r ? resultHTML(r, q) : ""}</div>
      </div>
    </div>`;
  const ta = $("#ans"), wc = $("#wc");
  const upd = () => { q._draft = ta.value; wc.textContent = `${ta.value.trim() ? ta.value.trim().split(/\s+/).length : 0} words`; };
  ta.oninput = upd; upd();
  $$(".qitem").forEach((b) => (b.onclick = () => { S.idx = +b.dataset.i; renderExam(); }));
  $("#prevQ").onclick = () => { S.idx--; renderExam(); };
  $("#nextQ").onclick = () => { S.idx++; renderExam(); };
  $("#submitQ").onclick = guard(async () => {
    if (!ta.value.trim()) return toast("Write an answer first.", true);
    const res = await api("/api/evaluate", { answer: ta.value, reference_answer: q.reference_answer, concepts: q.expected_concepts, keywords: q.expected_keywords, source_page: q.source_page, source_passage: q.source_passage }, "Evaluating with embeddings + NLI…");
    S.res[q.id] = { ...res, question_prompt: q.prompt, source_page: q.source_page, difficulty: q.difficulty, topic: q.topic };
    $$(".nav-item")[4].disabled = false;
    renderExam();
    animateRing();
  });
  animateRing();
}

function resultHTML(r, q) {
  const score = r.overall_score, C = 2 * Math.PI * 54, t = tone(score);
  const comp = (label, v) => `<div class="bar-row"><span class="lbl">${label}</span><div class="bar"><i style="width:${v}%"></i></div><span class="v">${pct(v)}</span></div>`;
  const ideas = (r.alignment || []).map((a) => {
    const conflict = (a.contradiction || 0) > 0.6, cls = conflict ? "conflict" : a.addressed ? "ok" : "no";
    return `<div class="idea ${cls}"><span class="ic">${conflict ? "!" : a.addressed ? "✓" : "✕"}</span><div>${esc(a.idea)}<em>${a.matched_sentence ? `Your closest sentence: “${esc(a.matched_sentence)}” (${a.similarity.toFixed(2)})` : "Nothing written about this."}${conflict ? " — <b>contradicts the material</b>" : ""}</em></div></div>`;
  }).join("");
  return `<div class="result">
    <div class="scoreRow">
      <div class="ring" data-score="${score}" data-c="${C}"><svg width="128" height="128"><circle class="track" cx="64" cy="64" r="54" fill="none" stroke-width="11"/><circle class="fg" cx="64" cy="64" r="54" fill="none" stroke-width="11" stroke-linecap="round" stroke="var(--${t})" stroke-dasharray="${C}" stroke-dashoffset="${C}"/></svg><div class="num">${Math.round(score)}</div></div>
      <div class="comps">${comp("Semantic similarity", r.semantic_similarity)}${comp("Concept coverage", r.concept_coverage)}${comp("Keyword coverage", r.keyword_coverage)}${r.contradiction_penalty ? `<div class="bar-row"><span class="lbl" style="color:var(--bad)">Contradiction penalty</span><div class="bar"><i style="width:${r.contradiction_penalty}%;background:var(--bad)"></i></div><span class="v">−${pct(r.contradiction_penalty)}</span></div>` : ""}</div>
    </div>
    <div class="alert ${r.contradictions?.length ? "bad" : t === "good" ? "good" : "info"}">${esc(r.feedback)}</div>
    ${(r.covered_concepts || []).length || (r.missing_concepts || []).length ? `<div class="chips">${r.covered_concepts.map((c) => `<span class="chip good">✓ ${esc(c)}</span>`).join("")}${r.missing_concepts.map((c) => `<span class="chip bad">✕ ${esc(c)}</span>`).join("")}</div>` : ""}
    ${ideas ? `<h3 style="margin-top:18px">Idea-by-idea check</h3>${ideas}` : ""}
    <details class="reveal"><summary>Show reference answer (page ${q.source_page})</summary><p>${esc(q.reference_answer)}</p></details>
  </div>`;
}
function animateRing() {
  const ring = $(".ring");
  if (!ring) return;
  const fg = $(".fg", ring), C = +ring.dataset.c, s = +ring.dataset.score;
  requestAnimationFrame(() => requestAnimationFrame(() => (fg.style.strokeDashoffset = C * (1 - Math.min(100, s) / 100))));
}

/* ---------- 5. results ---------- */
function renderResults() {
  const answers = S.qs.filter((q) => S.res[q.id]).map((q) => S.res[q.id]);
  const body = $("#resultsBody");
  if (!answers.length) { body.innerHTML = `<div class="card empty">Answer at least one question in the Exam to see results.</div>`; return; }
  const scores = answers.map((a) => a.overall_score), avg = scores.reduce((a, b) => a + b, 0) / scores.length;
  const conflicts = answers.reduce((n, a) => n + (a.contradictions?.length || 0), 0);
  body.innerHTML = `
    <div class="stats">
      <div class="stat"><b style="color:var(--${tone(avg)})">${pct(avg)}</b><span>Overall evaluation score</span></div>
      <div class="stat"><b>${pct(Math.max(...scores))}</b><span>Best answer</span></div>
      <div class="stat"><b>${answers.length}/${S.qs.length}</b><span>Questions answered</span></div>
      <div class="stat"><b>${conflicts}</b><span>Contradictions flagged</span></div>
    </div>
    <div class="grid two">
      <div class="card"><h3>Score per question</h3>${answers.map((a, i) => `<div class="bar-row"><span class="lbl">Q${i + 1}</span><div class="bar"><i style="width:${a.overall_score}%;background:var(--${tone(a.overall_score)})"></i></div><span class="v">${pct(a.overall_score)}</span></div>`).join("")}</div>
      <div class="card"><h3>Score components</h3><table><tr><th>Q</th><th>Semantic</th><th>Concepts</th><th>Keywords</th><th>Penalty</th></tr>${answers.map((a, i) => `<tr><td>${i + 1}</td><td>${pct(a.semantic_similarity)}</td><td>${pct(a.concept_coverage)}</td><td>${pct(a.keyword_coverage)}</td><td>${a.contradiction_penalty ? "−" + pct(a.contradiction_penalty) : "—"}</td></tr>`).join("")}</table></div>
    </div>
    <div class="card"><h3>📚 Revise these <span class="muted small">— retrieved from your document</span></h3><div id="plan" class="muted">Building revision plan…</div><div class="row"><button class="btn" id="dl">Download study report (.md)</button></div></div>
    <div class="card"><h3>Feedback by question</h3>${answers.map((a, i) => `<details><summary>Q${i + 1} · ${pct(a.overall_score)} · ${esc(a.question_prompt)}</summary><p>${esc(a.feedback)}</p></details>`).join("")}</div>`;
  guard(async () => {
    const rep = await api("/api/report", { filename: S.doc.filename, answers, chunks: S.doc.chunks }, "Building report…");
    $("#plan").innerHTML = rep.plan.length ? rep.plan.map((p) => `<div class="revise"><b>${esc(p.concept)}</b> · page ${p.page}${p.reason === "contradicted" ? ` <span class="chip bad">you contradicted this</span>` : ""}<blockquote>${esc(p.sentence)}</blockquote></div>`).join("") : "Nothing to revise — every reference concept was covered. 🎉";
    $("#dl").onclick = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([rep.markdown], { type: "text/markdown" })); a.download = "study_report.md"; a.click(); };
  })();
}

show("upload");
