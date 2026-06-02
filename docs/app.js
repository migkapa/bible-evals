/* eslint-disable no-restricted-globals */

function fmtPct(x) {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

function fmtNum(x) {
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(3);
}

// A table cell showing a point estimate with its 95% CI underneath (if present).
function ciCell(m, key, fmt) {
  const td = document.createElement("td");
  td.appendChild(el("div", { text: fmt(Number(m[key])) }));
  const c = ciFor(m, key);
  if (c) {
    td.appendChild(el("div", { class: "ci", text: `${fmt(c.lo)}–${fmt(c.hi)}` }));
  }
  return td;
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

function bestModels(history, key) {
  const best = new Map();
  for (const run of history) {
    for (const m of run.models || []) {
      if (isReferenceModel(m)) continue;
      const val = Number(m?.[key]);
      if (!Number.isFinite(val)) continue;
      const prev = best.get(m.model);
      if (!prev) {
        best.set(m.model, { ...m, _run_id: run.run_id, _created_at: run.created_at });
        continue;
      }
      const prevVal = Number(prev?.[key]);
      if (val > prevVal || (val === prevVal && String(run.run_id) > String(prev._run_id))) {
        best.set(m.model, { ...m, _run_id: run.run_id, _created_at: run.created_at });
      }
    }
  }
  return [...best.values()];
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
  const repro = run.repro || {};
  if (repro.git_commit) {
    rows.push(["commit", repro.git_dirty ? `${repro.git_commit} (dirty)` : repro.git_commit]);
  }
  if (repro.tool_version) rows.push(["tool", `v${repro.tool_version}`]);
  if (repro.python) rows.push(["python", repro.python]);
  if (repro.prompt_hash) rows.push(["prompt_hash", repro.prompt_hash]);

  for (const [k, v] of rows) {
    meta.appendChild(
      el("div", { class: "row" }, [
        el("div", { class: "key", text: k }),
        el("div", { class: "value", text: String(v) }),
      ]),
    );
  }

  if (repro.command) {
    const btn = el("button", { class: "button copy-btn", text: "Copy reproduce command" });
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(repro.command);
        const prev = btn.textContent;
        btn.textContent = "Copied ✓";
        setTimeout(() => {
          btn.textContent = prev;
        }, 1500);
      } catch (_) {
        btn.textContent = repro.command;
      }
    });
    meta.appendChild(el("div", { class: "row" }, [btn]));
  }
}

function ciFor(m, key) {
  const c = m && m.ci && m.ci[key];
  if (!c) return null;
  const lo = Number(c.lo);
  const hi = Number(c.hi);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  return { lo, hi };
}

function svgBarChart(models, key, { maxBars = 12 } = {}) {
  const sorted = [...models].sort(byKey(key, "desc")).slice(0, maxBars);
  const width = 980;
  const height = 260;
  const pad = { l: 40, r: 16, t: 14, b: 70 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  const vals = sorted.map((m) => Number(m[key]));
  // Extend the scale to the top of any CI whisker so bands aren't clipped.
  const ciHis = sorted.map((m) => (ciFor(m, key) || {}).hi).filter(Number.isFinite);
  const max = Math.max(1e-9, ...vals, ...ciHis);

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

    // Confidence-interval whisker (vertical line + caps) on top of the bar.
    const ci = ciFor(m, key);
    if (ci) {
      const cx = x + Math.max(1, barW - 12) / 2;
      const yHi = pad.t + innerH - (ci.hi / max) * innerH;
      const yLo = pad.t + innerH - (ci.lo / max) * innerH;
      const capW = Math.min(10, Math.max(4, (barW - 12) / 3));
      const whisk = document.createElementNS(ns, "g");
      whisk.setAttribute("stroke", "rgba(255,255,255,0.85)");
      whisk.setAttribute("stroke-width", "1.5");
      const seg = (x1, y1, x2, y2) => {
        const ln = document.createElementNS(ns, "line");
        ln.setAttribute("x1", String(x1));
        ln.setAttribute("y1", String(y1));
        ln.setAttribute("x2", String(x2));
        ln.setAttribute("y2", String(y2));
        whisk.appendChild(ln);
      };
      seg(cx, yHi, cx, yLo);
      seg(cx - capW / 2, yHi, cx + capW / 2, yHi);
      seg(cx - capW / 2, yLo, cx + capW / 2, yLo);
      const title = document.createElementNS(ns, "title");
      const fmt = key === "avg_wer" || key === "avg_cer" || key === "avg_chatter_ratio" || key === "avg_token_sort_ratio" ? fmtNum : fmtPct;
      title.textContent = `95% CI: ${fmt(ci.lo)} – ${fmt(ci.hi)}`;
      whisk.appendChild(title);
      g.appendChild(whisk);
    }

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
  const cis = points.map((p) => ciFor(p, key));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const domainYs = ys.filter((y) => Number.isFinite(y));
  for (const c of cis) {
    if (c) domainYs.push(c.lo, c.hi);
  }
  const finiteYs = domainYs;
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

  // Confidence band: filled area between per-point lo/hi, drawn behind the line.
  if (cis.some(Boolean)) {
    const upper = points.map((p, i) => {
      const c = cis[i];
      const y = c ? c.hi : Number(p[key]);
      return `${sx(p.t.getTime()).toFixed(2)} ${sy(y).toFixed(2)}`;
    });
    const lower = points
      .map((p, i) => {
        const c = cis[i];
        const y = c ? c.lo : Number(p[key]);
        return `${sx(p.t.getTime()).toFixed(2)} ${sy(y).toFixed(2)}`;
      })
      .reverse();
    const band = document.createElementNS(ns, "path");
    band.setAttribute("d", `M ${upper.join(" L ")} L ${lower.join(" L ")} Z`);
    band.setAttribute("fill", "rgba(34,197,94,0.16)");
    band.setAttribute("stroke", "none");
    svg.appendChild(band);
  }

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

function svgQuadrantChart(models) {
  const ns = "http://www.w3.org/2000/svg";
  const width = 980;
  const height = 460;
  const pad = { l: 56, r: 124, t: 20, b: 52 };
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");

  const sx = (v) => pad.l + Math.max(0, Math.min(1, v)) * innerW;
  const sy = (v) => pad.t + (1 - Math.max(0, Math.min(1, v))) * innerH;

  const mk = (tag, attrs) => {
    const n = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    return n;
  };

  const xm = sx(0.5);
  const ym = sy(0.5);

  // Quadrant background tints (only the "good" corner is emphasized).
  const quads = [
    { x: xm, y: pad.t, w: pad.l + innerW - xm, h: ym - pad.t, fill: "rgba(34,197,94,0.10)" },
    { x: pad.l, y: pad.t, w: xm - pad.l, h: ym - pad.t, fill: "rgba(96,165,250,0.06)" },
    { x: xm, y: ym, w: pad.l + innerW - xm, h: pad.t + innerH - ym, fill: "rgba(245,158,11,0.07)" },
    { x: pad.l, y: ym, w: xm - pad.l, h: pad.t + innerH - ym, fill: "rgba(239,68,68,0.07)" },
  ];
  for (const q of quads) {
    svg.appendChild(mk("rect", { x: q.x, y: q.y, width: q.w, height: q.h, fill: q.fill }));
  }

  // Plot border + midlines.
  svg.appendChild(
    mk("rect", {
      x: pad.l,
      y: pad.t,
      width: innerW,
      height: innerH,
      fill: "none",
      stroke: "rgba(255,255,255,0.14)",
    }),
  );
  for (const [x1, y1, x2, y2] of [
    [xm, pad.t, xm, pad.t + innerH],
    [pad.l, ym, pad.l + innerW, ym],
  ]) {
    const ln = mk("line", { x1, y1, x2, y2, stroke: "rgba(255,255,255,0.18)" });
    ln.setAttribute("stroke-dasharray", "4 5");
    svg.appendChild(ln);
  }

  // Corner labels.
  const corners = [
    { x: pad.l + innerW - 8, y: pad.t + 16, anchor: "end", text: "Verbatim machine" },
    { x: pad.l + 8, y: pad.t + 16, anchor: "start", text: "Refuses / concise miss" },
    { x: pad.l + innerW - 8, y: pad.t + innerH - 8, anchor: "end", text: "Knows but chatty" },
    { x: pad.l + 8, y: pad.t + innerH - 8, anchor: "start", text: "Confident but wrong" },
  ];
  for (const c of corners) {
    const t = mk("text", {
      x: c.x,
      y: c.y,
      fill: "rgba(255,255,255,0.40)",
      "font-size": "12",
      "text-anchor": c.anchor,
    });
    t.textContent = c.text;
    svg.appendChild(t);
  }

  // Axis ticks (0 / 50 / 100%).
  for (const v of [0, 0.5, 1]) {
    const xt = mk("text", {
      x: sx(v),
      y: pad.t + innerH + 18,
      fill: "rgba(255,255,255,0.6)",
      "font-size": "11",
      "text-anchor": "middle",
    });
    xt.textContent = fmtPct(v);
    svg.appendChild(xt);
    const yt = mk("text", {
      x: pad.l - 8,
      y: sy(v) + 4,
      fill: "rgba(255,255,255,0.6)",
      "font-size": "11",
      "text-anchor": "end",
    });
    yt.textContent = fmtPct(v);
    svg.appendChild(yt);
  }

  // Axis titles.
  const xTitle = mk("text", {
    x: pad.l + innerW / 2,
    y: height - 8,
    fill: "rgba(255,255,255,0.75)",
    "font-size": "12",
    "text-anchor": "middle",
  });
  xTitle.textContent = "Knows the verse  (content accuracy) →";
  svg.appendChild(xTitle);
  const yTitle = mk("text", {
    x: 16,
    y: pad.t + innerH / 2,
    fill: "rgba(255,255,255,0.75)",
    "font-size": "12",
    "text-anchor": "middle",
    transform: `rotate(-90 16 ${pad.t + innerH / 2})`,
  });
  yTitle.textContent = "Says only the verse  (clean output) →";
  svg.appendChild(yTitle);

  const pts = models
    .filter((m) => !isReferenceModel(m))
    .map((m) => ({ m, x: Number(m.content_accuracy), y: Number(m.clean_output_rate) }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));

  if (!pts.length) {
    const t = mk("text", {
      x: pad.l + innerW / 2,
      y: pad.t + innerH / 2,
      fill: "rgba(255,255,255,0.6)",
      "font-size": "13",
      "text-anchor": "middle",
    });
    t.textContent = "No model data available.";
    svg.appendChild(t);
    return svg;
  }

  const dotColor = (x, y) => {
    if (x >= 0.5 && y >= 0.5) return "rgba(34,197,94,0.95)";
    if (x >= 0.5 && y < 0.5) return "rgba(245,158,11,0.95)";
    if (x < 0.5 && y >= 0.5) return "rgba(96,165,250,0.95)";
    return "rgba(239,68,68,0.95)";
  };

  // Draw dots first.
  for (const p of pts) {
    const dot = mk("circle", {
      cx: sx(p.x),
      cy: sy(p.y),
      r: 7,
      fill: dotColor(p.x, p.y),
      stroke: "rgba(0,0,0,0.45)",
      "stroke-width": "1.5",
    });
    const title = document.createElementNS(ns, "title");
    title.textContent = `${p.m.model}\ncontent accuracy: ${fmtPct(p.x)}\nclean output: ${fmtPct(p.y)}`;
    dot.appendChild(title);
    svg.appendChild(dot);
  }

  // Place labels with greedy vertical de-collision so dots sharing a score
  // (common at 100% clean output) don't print on top of each other.
  const placed = [];
  const top = pad.t + 6;
  const bottom = pad.t + innerH - 4;
  for (const p of [...pts].sort((a, b) => sy(a.y) - sy(b.y) || sx(a.x) - sx(b.x))) {
    const cx = sx(p.x);
    const nearRight = cx > pad.l + innerW - 90;
    const anchor = nearRight ? "end" : "start";
    const lx = nearRight ? cx - 11 : cx + 11;
    let ly = sy(p.y) + 4;
    while (
      placed.some((q) => q.anchor === anchor && Math.abs(q.lx - lx) < 96 && Math.abs(q.ly - ly) < 14)
    ) {
      ly += 14;
      if (ly > bottom) {
        ly = top;
        break;
      }
    }
    placed.push({ lx, ly, anchor });
    const label = mk("text", {
      x: lx,
      y: ly,
      fill: "rgba(255,255,255,0.9)",
      "font-size": "12",
      "text-anchor": anchor,
    });
    label.textContent = String(p.m.model).replace(/^ollama:/, "");
    svg.appendChild(label);
  }

  return svg;
}

function renderLeaderboardTable(models) {
  const table = document.getElementById("leaderboard");
  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";
  for (const m of models) {
    const tr = document.createElement("tr");
    tr.appendChild(el("td", {}, [el("span", { class: "pill", text: m.model })]));
    tr.appendChild(ciCell(m, "strict_accuracy", fmtPct));
    tr.appendChild(ciCell(m, "content_accuracy", fmtPct));
    tr.appendChild(ciCell(m, "clean_output_rate", fmtPct));
    tr.appendChild(ciCell(m, "avg_wer", fmtNum));
    tr.appendChild(ciCell(m, "avg_cer", fmtNum));
    tr.appendChild(ciCell(m, "avg_token_sort_ratio", fmtNum));
    tr.appendChild(ciCell(m, "avg_chatter_ratio", fmtNum));
    tr.appendChild(el("td", { text: String(m.n ?? "—") }));
    tbody.appendChild(tr);
  }
}

function renderModelCards(models) {
  const host = document.getElementById("modelCards");
  host.innerHTML = "";
  const ordered = [...models].sort(byKey("strict_accuracy", "desc"));

  for (const m of ordered) {
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

    const ab = m.abstention;
    if (ab && Number.isFinite(Number(ab.n)) && Number(ab.n) > 0) {
      const refused = Number(ab.refused);
      const rate = Number(ab.abstention_rate);
      card.querySelector(".kpis").appendChild(
        el("div", { class: "kpi" }, [
          el("div", { class: "k", text: "Refused fake refs" }),
          el("div", {
            class: "v",
            text: `${refused}/${ab.n} (${fmtPct(rate)})`,
          }),
        ]),
      );
    }

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

function wireTableSorting(models) {
  const table = document.getElementById("leaderboard");
  const headers = [...table.querySelectorAll("thead th[data-key]")];
  let sortKey = "strict_accuracy";
  let sortDir = "desc";

  function apply() {
    const ordered = [...models].sort(byKey(sortKey, sortDir));
    renderLeaderboardTable(ordered);
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

const LABEL_COLORS = {
  verbatim: "rgba(34,197,94,0.92)",
  verbatim_with_extras: "rgba(96,165,250,0.92)",
  inaccurate_recall: "rgba(245,158,11,0.92)",
  total_hallucination: "rgba(239,68,68,0.92)",
};
const LABEL_TEXT = {
  verbatim: "verbatim",
  verbatim_with_extras: "verbatim + extras",
  inaccurate_recall: "inaccurate recall",
  total_hallucination: "off-target",
};

async function renderHeatmap(latest) {
  const host = document.getElementById("heatmap");
  if (!host) return;
  host.textContent = "Loading heatmap…";

  const models = [...(latest.models || [])]
    .filter((m) => !isReferenceModel(m))
    .sort(byKey("strict_accuracy", "desc"));
  if (!models.length) {
    host.textContent = "No non-reference models in the latest run.";
    return;
  }

  const perModel = await Promise.all(
    models.map(async (m) => {
      try {
        return await loadDetailsForModel(latest, m.model_slug || m.model);
      } catch (_) {
        return null;
      }
    }),
  );

  // Verse rows: preserve sample order from the first model that loaded.
  const order = [];
  const seen = new Set();
  const byRefByModel = models.map(() => new Map());
  perModel.forEach((entries, ci) => {
    if (!Array.isArray(entries)) return;
    for (const e of entries) {
      const v = e.verse || {};
      const ref = v.ref || `${v.book ?? "?"} ${v.chapter ?? "?"}:${v.verse ?? "?"}`;
      if (!seen.has(ref)) {
        seen.add(ref);
        order.push(ref);
      }
      byRefByModel[ci].set(ref, e.scores || {});
    }
  });

  if (!order.length) {
    host.textContent = "No detailed results available for the latest run.";
    return;
  }

  const ns = "http://www.w3.org/2000/svg";
  const leftPad = 150;
  const topPad = 96;
  const rowH = 22;
  const width = 980;
  const colW = Math.max(34, (width - leftPad - 12) / models.length);
  const gridW = colW * models.length;
  const height = topPad + order.length * rowH + 40;

  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");

  const mk = (tag, attrs, text) => {
    const n = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
    if (text != null) n.textContent = text;
    return n;
  };

  // Column headers (model names), rotated to avoid overlap.
  models.forEach((m, ci) => {
    const x = leftPad + ci * colW + colW / 2;
    const t = mk(
      "text",
      {
        x,
        y: topPad - 10,
        fill: "rgba(255,255,255,0.85)",
        "font-size": "12",
        "text-anchor": "start",
        transform: `rotate(-40 ${x} ${topPad - 10})`,
      },
      String(m.model).replace(/^ollama:/, ""),
    );
    svg.appendChild(t);
  });

  // Cells.
  order.forEach((ref, ri) => {
    const y = topPad + ri * rowH;
    const rowLabel = mk(
      "text",
      {
        x: leftPad - 8,
        y: y + rowH / 2 + 4,
        fill: "rgba(255,255,255,0.82)",
        "font-size": "12",
        "text-anchor": "end",
      },
      ref,
    );
    svg.appendChild(rowLabel);

    models.forEach((m, ci) => {
      const x = leftPad + ci * colW;
      const sc = byRefByModel[ci].get(ref);
      const label = sc ? sc.label : null;
      const fill = (label && LABEL_COLORS[label]) || "rgba(255,255,255,0.06)";
      const cell = mk("rect", {
        x: x + 2,
        y: y + 2,
        width: colW - 4,
        height: rowH - 4,
        rx: 4,
        fill,
        stroke: "rgba(0,0,0,0.35)",
        "stroke-width": "1",
      });
      const wer = sc ? Number(sc.wer) : NaN;
      const title = document.createElementNS(ns, "title");
      title.textContent = `${m.model}\n${ref}\n${
        label ? LABEL_TEXT[label] || label : "no data"
      }${Number.isFinite(wer) ? `\nWER ${wer.toFixed(3)}` : ""}`;
      cell.appendChild(title);
      svg.appendChild(cell);
    });
  });

  // Legend.
  let lx = leftPad;
  const ly = topPad + order.length * rowH + 22;
  for (const key of Object.keys(LABEL_COLORS)) {
    svg.appendChild(
      mk("rect", { x: lx, y: ly - 11, width: 14, height: 14, rx: 3, fill: LABEL_COLORS[key] }),
    );
    const t = mk(
      "text",
      { x: lx + 20, y: ly, fill: "rgba(255,255,255,0.78)", "font-size": "12" },
      LABEL_TEXT[key],
    );
    svg.appendChild(t);
    lx += 24 + LABEL_TEXT[key].length * 7.5 + 18;
  }

  host.innerHTML = "";
  host.appendChild(svg);
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

  const models = bestModels(history, "strict_accuracy").sort(byKey("strict_accuracy", "desc"));
  const barChart = document.getElementById("barChart");
  barChart.innerHTML = "";
  barChart.appendChild(
    svgBarChart(
      models.filter((m) => !isReferenceModel(m)),
      "strict_accuracy",
    ),
  );

  renderModelCards(models);

  const quadrant = document.getElementById("quadrantChart");
  if (quadrant) {
    quadrant.innerHTML = "";
    quadrant.appendChild(svgQuadrantChart(models));
  }

  wireTableSorting(models);
  wireTrend(history);
  renderHeatmap(latest);
  wireExamples(latest);
}

main().catch((err) => {
  console.error(err);
  document.getElementById("runMeta").textContent = `Failed to load data: ${String(err)}`;
});
