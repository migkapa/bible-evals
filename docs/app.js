/* eslint-disable no-restricted-globals */

function fmtPct(x) {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

function fmtNum(x) {
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(3);
}

function byKey(key, dir = "desc") {
  const sign = dir === "asc" ? 1 : -1;
  return (a, b) => {
    const av = a[key];
    const bv = b[key];
    if (typeof av === "string" || typeof bv === "string") {
      return sign * String(av).localeCompare(String(bv));
    }
    const an = Number(av);
    const bn = Number(bv);
    if (!Number.isFinite(an) && !Number.isFinite(bn)) return 0;
    if (!Number.isFinite(an)) return 1;
    if (!Number.isFinite(bn)) return -1;
    return sign * (an - bn);
  };
}

function latestRun(history) {
  if (!history.length) return null;
  return [...history].sort((a, b) => String(a.run_id).localeCompare(String(b.run_id))).at(-1);
}

function uniqueModels(history) {
  const set = new Set();
  for (const run of history) {
    for (const m of run.models || []) {
      if (isReferenceModel(m)) continue;
      set.add(m.model);
    }
  }
  return [...set].sort();
}

function isReferenceModel(m) {
  const name = String(m?.model || "");
  const slug = String(m?.model_slug || "");
  return name.startsWith("reference:") || slug.startsWith("reference_");
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.appendChild(c);
  return node;
}

function tokenizeWords(s) {
  const out = String(s || "")
    .trim()
    .split(/\s+/g)
    .filter(Boolean);
  return out;
}

function diffOps(refTokens, hypTokens) {
  const n = refTokens.length;
  const m = hypTokens.length;
  const dp = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0));

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      dp[i][j] =
        refTokens[i - 1] === hypTokens[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  let i = n;
  let j = m;
  const ops = [];
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && refTokens[i - 1] === hypTokens[j - 1]) {
      ops.push({ op: "eq", t: refTokens[i - 1] });
      i--;
      j--;
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      ops.push({ op: "ins", t: hypTokens[j - 1] });
      j--;
    } else {
      ops.push({ op: "del", t: refTokens[i - 1] });
      i--;
    }
  }

  ops.reverse();
  return ops;
}

function renderDiff(ref, hyp) {
  const ops = diffOps(tokenizeWords(ref), tokenizeWords(hyp));
  const gt = document.createElement("div");
  const pr = document.createElement("div");

  for (const o of ops) {
    if (o.op === "eq") {
      gt.appendChild(document.createTextNode(`${o.t} `));
      pr.appendChild(document.createTextNode(`${o.t} `));
    } else if (o.op === "del") {
      const s = el("span", { class: "del", text: o.t });
      gt.appendChild(s);
      gt.appendChild(document.createTextNode(" "));
    } else if (o.op === "ins") {
      const s = el("span", { class: "ins", text: o.t });
      pr.appendChild(s);
      pr.appendChild(document.createTextNode(" "));
    }
  }

  const host = el("div", { class: "diff" }, [
    el("div", { class: "row" }, [el("div", { class: "tag", text: "GT (−)" }), gt]),
    el("div", { class: "row" }, [el("div", { class: "tag", text: "Pred (+)" }), pr]),
  ]);
  return host;
}

function renderMeta(run) {
  const meta = document.getElementById("runMeta");
  meta.innerHTML = "";
  const rows = [
    ["run_id", run.run_id],
    ["created_at", run.created_at || run.run_id],
    ["version", run.version],
    ["prompt_mode", run.prompt_mode],
    ["models", String((run.models || []).length)],
    ["sample", JSON.stringify(run.sample || {})],
  ];
  for (const [k, v] of rows) {
    meta.appendChild(
      el("div", { class: "row" }, [
        el("div", { class: "key", text: k }),
        el("div", { class: "value", text: String(v) }),
      ]),
    );
  }
}

function svgBarChart(models, key, { maxBars = 12 } = {}) {
  const sorted = [...models].sort(byKey(key, "desc")).slice(0, maxBars);
  const width = 980;
  const height = 260;
  const pad = { l: 40, r: 16, t: 14, b: 70 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  const vals = sorted.map((m) => Number(m[key]));
  const max = Math.max(1e-9, ...vals);

  const barW = innerW / Math.max(1, sorted.length);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");

  const grid = document.createElementNS(ns, "g");
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (innerH * i) / 4;
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", String(pad.l));
    line.setAttribute("x2", String(width - pad.r));
    line.setAttribute("y1", String(y));
    line.setAttribute("y2", String(y));
    line.setAttribute("stroke", "rgba(255,255,255,0.10)");
    grid.appendChild(line);
  }
  svg.appendChild(grid);

  const g = document.createElementNS(ns, "g");
  sorted.forEach((m, i) => {
    const v = Number(m[key]);
    const x = pad.l + i * barW + 6;
    const h = Math.max(0, (v / max) * innerH);
    const y = pad.t + (innerH - h);
    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("x", String(x));
    rect.setAttribute("y", String(y));
    rect.setAttribute("width", String(Math.max(1, barW - 12)));
    rect.setAttribute("height", String(h));
    rect.setAttribute("rx", "8");
    let fill = "rgba(124, 58, 237, 0.85)";
    if (key === "strict_accuracy") {
      if (v >= 0.85) fill = "rgba(34, 197, 94, 0.86)";
      else if (v >= 0.6) fill = "rgba(245, 158, 11, 0.86)";
      else fill = "rgba(239, 68, 68, 0.86)";
    }
    rect.setAttribute("fill", fill);
    rect.setAttribute("stroke", "rgba(255,255,255,0.14)");
    rect.setAttribute("stroke-width", "1");
    g.appendChild(rect);

    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", String(x + Math.max(1, barW - 12) / 2));
    label.setAttribute("y", String(height - 18));
    label.setAttribute("fill", "rgba(255,255,255,0.78)");
    label.setAttribute("font-size", "12");
    label.setAttribute("text-anchor", "middle");
    label.textContent = m.model.replace(/^ollama:/, "");
    g.appendChild(label);

    const val = document.createElementNS(ns, "text");
    val.setAttribute("x", String(x + Math.max(1, barW - 12) / 2));
    val.setAttribute("y", String(y - 6));
    val.setAttribute("fill", "rgba(255,255,255,0.86)");
    val.setAttribute("font-size", "12");
    val.setAttribute("text-anchor", "middle");
    val.textContent = key === "strict_accuracy" ? fmtPct(v) : fmtNum(v);
    g.appendChild(val);
  });
  svg.appendChild(g);
  return svg;
}

function svgLineChart(points, key) {
  const width = 980;
  const height = 260;
  const pad = { l: 46, r: 16, t: 14, b: 34 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  const xs = points.map((p) => p.t.getTime());
  const ys = points.map((p) => Number(p[key]));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const finiteYs = ys.filter((y) => Number.isFinite(y));
  const minY = finiteYs.length ? Math.min(...finiteYs) : 0;
  const maxY = finiteYs.length ? Math.max(...finiteYs) : 1;
  const yPad = (maxY - minY) * 0.08 || 0.1;
  const y0 = minY - yPad;
  const y1 = maxY + yPad;

  const sx = (x) => pad.l + ((x - minX) / Math.max(1, maxX - minX)) * innerW;
  const sy = (y) => pad.t + (1 - (y - y0) / Math.max(1e-9, y1 - y0)) * innerH;

  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");

  const grid = document.createElementNS(ns, "g");
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (innerH * i) / 4;
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", String(pad.l));
    line.setAttribute("x2", String(width - pad.r));
    line.setAttribute("y1", String(y));
    line.setAttribute("y2", String(y));
    line.setAttribute("stroke", "rgba(255,255,255,0.10)");
    grid.appendChild(line);

    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", String(pad.l - 10));
    label.setAttribute("y", String(y + 4));
    label.setAttribute("fill", "rgba(255,255,255,0.65)");
    label.setAttribute("font-size", "12");
    label.setAttribute("text-anchor", "end");
    const val = y0 + ((y1 - y0) * (4 - i)) / 4;
    label.textContent = key === "strict_accuracy" ? fmtPct(val) : fmtNum(val);
    grid.appendChild(label);
  }
  svg.appendChild(grid);

  const path = document.createElementNS(ns, "path");
  const d = points
    .map((p, i) => {
      const x = sx(p.t.getTime());
      const y = sy(Number(p[key]));
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  path.setAttribute("d", d);
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "rgba(34,197,94,0.9)");
  path.setAttribute("stroke-width", "3");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  svg.appendChild(path);

  const dots = document.createElementNS(ns, "g");
  points.forEach((p) => {
    const cx = sx(p.t.getTime());
    const cy = sy(Number(p[key]));
    const c = document.createElementNS(ns, "circle");
    c.setAttribute("cx", String(cx));
    c.setAttribute("cy", String(cy));
    c.setAttribute("r", "5");
    c.setAttribute("fill", "rgba(34,197,94,0.95)");
    c.setAttribute("stroke", "rgba(255,255,255,0.20)");
    c.setAttribute("stroke-width", "1");
    dots.appendChild(c);
  });
  svg.appendChild(dots);

  return svg;
}

function renderLeaderboardTable(models) {
  const table = document.getElementById("leaderboard");
  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";
  for (const m of models) {
    const tr = document.createElement("tr");
    tr.appendChild(el("td", {}, [el("span", { class: "pill", text: m.model })]));
    tr.appendChild(el("td", { text: fmtPct(Number(m.strict_accuracy)) }));
    tr.appendChild(el("td", { text: fmtPct(Number(m.content_accuracy)) }));
    tr.appendChild(el("td", { text: fmtPct(Number(m.clean_output_rate)) }));
    tr.appendChild(el("td", { text: fmtNum(Number(m.avg_wer)) }));
    tr.appendChild(el("td", { text: fmtNum(Number(m.avg_cer)) }));
    tr.appendChild(el("td", { text: fmtNum(Number(m.avg_token_sort_ratio)) }));
    tr.appendChild(el("td", { text: fmtNum(Number(m.avg_chatter_ratio)) }));
    tr.appendChild(el("td", { text: String(m.n ?? "—") }));
    tbody.appendChild(tr);
  }
}

function renderModelCards(latest) {
  const host = document.getElementById("modelCards");
  host.innerHTML = "";
  const models = [...(latest.models || [])]
    .filter((m) => !isReferenceModel(m))
    .sort(byKey("strict_accuracy", "desc"));

  for (const m of models) {
    const grade = String(m.grade || "—");
    const headline = String(m.headline || "");
    const notes = Array.isArray(m.notes) ? m.notes : [];

    const exact = Number(m.verbatim_count ?? NaN);
    const exactPlus = Number(m.verbatim_with_extras_count ?? NaN);
    const halluc = Number(m.hallucination_count ?? NaN);
    const inaccurate = Number(m.inaccurate_count ?? NaN);
    const n = Number(m.n ?? NaN);

    const card = el("div", { class: "model-card" }, [
      el("div", { class: "top" }, [
        el("div", { class: "name", text: m.model }),
        el("div", { class: `grade ${grade}` }, [
          el("span", { text: "Grade" }),
          el("span", { text: grade }),
        ]),
      ]),
      el("div", { class: "headline", text: headline || "—" }),
      el("div", { class: "kpis" }, [
        el("div", { class: "kpi" }, [
          el("div", { class: "k", text: "Exact quotes" }),
          el("div", {
            class: "v",
            text: Number.isFinite(exact) && Number.isFinite(n) ? `${exact}/${n}` : "—",
          }),
        ]),
        el("div", { class: "kpi" }, [
          el("div", { class: "k", text: "Correct (w/ extras)" }),
          el("div", {
            class: "v",
            text: Number.isFinite(exactPlus) && Number.isFinite(n) ? `${exactPlus}/${n}` : "—",
          }),
        ]),
        el("div", { class: "kpi" }, [
          el("div", { class: "k", text: "Off-target" }),
          el("div", {
            class: "v",
            text: Number.isFinite(halluc) && Number.isFinite(n) ? `${halluc}/${n}` : "—",
          }),
        ]),
        el("div", { class: "kpi" }, [
          el("div", { class: "k", text: "Content accuracy" }),
          el("div", { class: "v", text: fmtPct(Number(m.content_accuracy)) }),
        ]),
        el("div", { class: "kpi" }, [
          el("div", { class: "k", text: "Clean output" }),
          el("div", { class: "v", text: fmtPct(Number(m.clean_output_rate)) }),
        ]),
      ]),
    ]);

    if (notes.length) {
      const ul = document.createElement("ul");
      ul.className = "notes";
      for (const t of notes.slice(0, 4)) ul.appendChild(el("li", { text: String(t) }));
      card.appendChild(ul);
    } else if (Number.isFinite(inaccurate) && Number.isFinite(n)) {
      const ul = document.createElement("ul");
      ul.className = "notes";
      ul.appendChild(el("li", { text: `${inaccurate}/${n} were close-but-not-verbatim.` }));
      card.appendChild(ul);
    }

    host.appendChild(card);
  }
}

async function loadDetailsForModel(latest, modelSlugOrName) {
  const m =
    (latest.models || []).find((x) => x.model_slug === modelSlugOrName) ||
    (latest.models || []).find((x) => x.model === modelSlugOrName);
  if (!m) return null;
  const rel = m.details_rel;
  if (!rel) return null;
  const res = await fetch(`./data/${rel}`, { cache: "no-store" });
  return res.json();
}

function renderExamples(latest, entries, model, kind, count) {
  const host = document.getElementById("examples");
  host.innerHTML = "";
  if (!Array.isArray(entries) || entries.length === 0) {
    host.textContent = "No detailed results available for examples.";
    return;
  }

  const c = Number(count) || 5;
  let list = [];
  if (kind === "best") {
    list = entries.filter((e) => e.scores?.label === "verbatim").slice(0, c);
  } else if (kind === "verbatim_with_extras") {
    list = entries.filter((e) => e.scores?.label === "verbatim_with_extras").slice(0, c);
  } else if (kind === "hallucinations") {
    list = entries.filter((e) => e.scores?.label === "total_hallucination").slice(0, c);
  } else if (kind === "truncated") {
    list = entries
      .filter((e) => Number(e.scores?.chatter_ratio ?? 0) < -0.15)
      .sort((a, b) => Number(a.scores?.chatter_ratio ?? 0) - Number(b.scores?.chatter_ratio ?? 0))
      .slice(0, c);
  } else if (kind === "chattery") {
    list = entries
      .filter((e) => Number(e.scores?.chatter_ratio ?? 0) > 0.15)
      .sort((a, b) => Number(b.scores?.chatter_ratio ?? 0) - Number(a.scores?.chatter_ratio ?? 0))
      .slice(0, c);
  } else {
    list = [...entries]
      .filter((e) => e.scores?.label !== "verbatim")
      .sort((a, b) => Number(b.scores?.wer ?? 0) - Number(a.scores?.wer ?? 0))
      .slice(0, c);
  }

  const title = el("div", {
    class: "hint",
    text: `Showing ${list.length} example(s) for ${model.model}`,
  });
  host.appendChild(title);

  for (const e of list) {
    const ref = e.verse?.ref || `${e.verse?.book ?? "?"} ${e.verse?.chapter ?? "?"}:${e.verse?.verse ?? "?"}`;
    const label = String(e.scores?.label || "unknown");
    const wer = Number(e.scores?.wer ?? NaN);
    const tsr = Number(e.scores?.token_sort_ratio ?? NaN);
    const cer = Number(e.scores?.cer ?? NaN);
    const chatter = Number(e.scores?.chatter_ratio ?? NaN);

    const d = document.createElement("details");
    d.className = "ex";
    d.appendChild(
      el("summary", {
        text: `${ref} • ${label} • WER ${Number.isFinite(wer) ? wer.toFixed(3) : "—"} • CER ${
          Number.isFinite(cer) ? cer.toFixed(3) : "—"
        } • Chatter ${Number.isFinite(chatter) ? chatter.toFixed(3) : "—"} • Fuzzy ${
          Number.isFinite(tsr) ? tsr.toFixed(1) : "—"
        }`,
      }),
    );

    const meta = el("div", { class: "meta2" }, [
      el("span", { text: `model=${model.model}` }),
      el("span", { text: `prompt=${latest.prompt_mode}` }),
      el("span", { text: `version=${latest.version}` }),
    ]);
    d.appendChild(meta);

    const prompt = e.prompt || {};
    const promptSystem = prompt.system;
    const promptUser = prompt.user;
    if (promptSystem || promptUser) {
      const pd = document.createElement("details");
      pd.className = "prompt";
      pd.appendChild(el("summary", { text: "Prompt used" }));
      if (promptSystem) {
        pd.appendChild(el("div", { class: "hint", text: "System:" }));
        pd.appendChild(el("pre", { text: String(promptSystem) }));
      }
      if (promptUser) {
        pd.appendChild(el("div", { class: "hint", text: "User:" }));
        pd.appendChild(el("pre", { text: String(promptUser) }));
      }
      d.appendChild(pd);
    }

    d.appendChild(el("div", { class: "hint", text: "Ground truth:" }));
    d.appendChild(el("pre", { text: String(e.verse?.text || "") }));
    d.appendChild(el("div", { class: "hint", text: "Model output (scored):" }));
    d.appendChild(el("pre", { text: String(e.prediction || "") }));
    if (e.prediction_raw && e.prediction_raw !== e.prediction) {
      const rd = document.createElement("details");
      rd.className = "prompt";
      rd.appendChild(el("summary", { text: "Raw model output" }));
      rd.appendChild(el("pre", { text: String(e.prediction_raw) }));
      d.appendChild(rd);
    }
    const pp = e.postprocess || {};
    if (pp.strip_thinking) {
      const changed = pp.strip_thinking_changed ? "yes" : "no";
      d.appendChild(el("div", { class: "meta2" }, [el("span", { text: `postprocess=strip_thinking changed=${changed}` })]));
    }
    d.appendChild(renderDiff(String(e.verse?.text || ""), String(e.prediction || "")));

    host.appendChild(d);
  }
}

function wireTableSorting(latest) {
  const table = document.getElementById("leaderboard");
  const headers = [...table.querySelectorAll("thead th[data-key]")];
  let sortKey = "strict_accuracy";
  let sortDir = "desc";

  function apply() {
    const models = [...(latest.models || [])]
      .filter((m) => !isReferenceModel(m))
      .sort(byKey(sortKey, sortDir));
    renderLeaderboardTable(models);
  }

  headers.forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.getAttribute("data-key");
      if (!key) return;
      if (sortKey === key) sortDir = sortDir === "desc" ? "asc" : "desc";
      else {
        sortKey = key;
        sortDir = key === "model" ? "asc" : "desc";
      }
      apply();
    });
  });

  apply();
}

function wireTrend(history) {
  const modelSelect = document.getElementById("modelSelect");
  const metricSelect = document.getElementById("metricSelect");
  const trendChart = document.getElementById("trendChart");

  const models = uniqueModels(history);
  modelSelect.innerHTML = "";
  for (const m of models) modelSelect.appendChild(el("option", { value: m, text: m }));

  function render() {
    const model = modelSelect.value;
    const metric = metricSelect.value;
    const points = [];
    for (const run of history) {
      const t = new Date(run.created_at || run.run_id);
      const mm = (run.models || []).find((x) => x.model === model);
      if (!mm) continue;
      points.push({ t, ...mm });
    }
    points.sort((a, b) => a.t.getTime() - b.t.getTime());
    trendChart.innerHTML = "";
    if (points.length < 1) {
      trendChart.textContent = "No data for selection.";
      return;
    }
    trendChart.appendChild(svgLineChart(points, metric));
  }

  modelSelect.addEventListener("change", render);
  metricSelect.addEventListener("change", render);
  if (models.length) {
    modelSelect.value = models[0];
    render();
  } else {
    trendChart.textContent = "No non-reference models available.";
  }
}

function wireExamples(latest) {
  const select = document.getElementById("exampleModelSelect");
  const kindSelect = document.getElementById("exampleKindSelect");
  const countSelect = document.getElementById("exampleCountSelect");
  const host = document.getElementById("examples");
  const models = [...(latest.models || [])]
    .filter((m) => !isReferenceModel(m))
    .sort(byKey("strict_accuracy", "desc"));
  select.innerHTML = "";
  for (const m of models) {
    select.appendChild(el("option", { value: m.model_slug || m.model, text: m.model }));
  }

  async function render() {
    host.textContent = "Loading examples…";
    const key = select.value;
    const chosen = models.find((m) => (m.model_slug || m.model) === key) || models[0];
    try {
      const entries = await loadDetailsForModel(latest, key);
      renderExamples(latest, entries, chosen, kindSelect.value, countSelect.value);
    } catch (e) {
      host.textContent = `Failed to load examples: ${String(e)}`;
    }
  }

  select.addEventListener("change", render);
  kindSelect.addEventListener("change", render);
  countSelect.addEventListener("change", render);
  if (models.length) {
    select.value = models[0].model_slug || models[0].model;
    if (countSelect) countSelect.value = "5";
    render();
  } else {
    host.textContent = "No non-reference models available.";
  }
}

async function main() {
  const res = await fetch("./data/history.json", { cache: "no-store" });
  const history = await res.json();
  if (!Array.isArray(history) || history.length === 0) {
    document.getElementById("runMeta").textContent =
      "No results yet. Run: bible-eval run --config config.yaml";
    return;
  }

  const latest = latestRun(history);
  renderMeta(latest);

  const models = [...(latest.models || [])].sort(byKey("strict_accuracy", "desc"));
  const barChart = document.getElementById("barChart");
  barChart.innerHTML = "";
  barChart.appendChild(
    svgBarChart(
      models.filter((m) => !isReferenceModel(m)),
      "strict_accuracy",
    ),
  );

  renderModelCards(latest);
  wireTableSorting(latest);
  wireTrend(history);
  wireExamples(latest);
}

main().catch((err) => {
  console.error(err);
  document.getElementById("runMeta").textContent = `Failed to load data: ${String(err)}`;
});
