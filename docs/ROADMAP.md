# BibleEval Roadmap: Cooler Things to Build

*Synthesized from web research (memorization science, scripture-NLP, leaderboard platforms, metrics methodology, interactive viz, adjacent ideas) and a full codebase audit. Opinionated. Sequenced. File paths are absolute.*

## Shipped

- ✅ **Statistical honesty** — Wilson + bootstrap 95% CIs on every metric; reproduce stamp (commit, seed, prompt hash). Bar-chart whiskers, trend noise band, leaderboard CI sub-rows.
- ✅ **Knowledge-vs-Compliance quadrant** — content accuracy × clean-output rate scatter with the four named quadrants.
- ✅ **Abstention probes** — non-existent references (e.g. `John 999:1`); rewards refusal, flags fabrication. Surfaced as a model-card KPI.
- ✅ **Verse × Model heatmap** — per-verse label grid across models for the latest run.
- ✅ **Engineering health** — GitHub Actions CI (ruff + pytest, Py 3.9–3.12), declared `rapidfuzz` as a `[fuzzy]`/`[dev]` extra, dropped unused `tqdm`, real `[dev]` extra.

---

## 1. Executive Summary — the 5 highest-leverage bets

The project's stated north star is **separating KNOWLEDGE from COMPLIANCE**, but the current pipeline only half-instruments it: it measures *whether words are right* (WER/CER) but never *whether the model knew it was right*, *what kind of non-compliance occurred*, or *whether a "0.80" is even real at N=10*. The coolest, most defensible upgrades all push on that gap — and most are cheap because the per-verse details JSON already shipped to `docs/data/details/` is far richer than the site currently uses.

**My top 5, in priority order:**

1. **Ship statistical honesty first (Wilson + bootstrap CIs).** Every serious eval (HELM, Inspect, lm-eval-harness, LMSYS) ships uncertainty with every number; we report bare point estimates over N=10. ~30 lines of pure Python in `cli.py`, zero new deps, and it instantly reframes the whole leaderboard from overconfident to credible. **Do this first.**

2. **The Knowledge-vs-Compliance 2×2 quadrant chart.** The one visualization that *is* the project's thesis: x = content_accuracy (knows it), y = clean_output_rate (says only it). Four quadrants, one dot per model. All fields already exist; it's pure front-end SVG.

3. **Abstention / fake-reference probes.** Inject non-existent references ("John 99:1", "Obadiah 2:5") into the sampler and reward refusal over fabrication. The *Sacred or Synthetic?* literature names abstention as **THE** discriminating axis for religious-domain reliability — and we currently have no concept of "should not answer." Near-free, genuinely missing safety dimension.

4. **Verse × Model accuracy heatmap.** Turn 14 separate per-model detail files into one at-a-glance grid (rows = verses, cols = models, color = label). Reveals which verses are *universally* hard — i.e. the popularity/memorization structure that is the real phenomenon under study. Highest signal-per-pixel.

5. **Popularity-stratified "memorization gradient."** Lean into contamination instead of fighting it: tier verses famous→obscure, report `accuracy(famous) − accuracy(obscure)` per model. This reframes an apparent weakness (these texts are in every training set) as the *headline scientific finding* and aligns us with the memorization-extraction literature.

**Strategic framing:** position BibleEval as the open **Bible counterpart to IslamicEval 2025 / IslamicMMLU** — an active, citable niche with *no published Bible equivalent*. That framing makes the dataset release and the famous-vs-obscure gradient publishable, not just cute.

---

## 2. Quick wins — shippable this week

### Statistics & credibility
- **CI Whiskers** — *Wilson score intervals on every rate metric.* Render `0.80 [0.49–0.94]` in the leaderboard and as error bars on the SVG bars. Why cool: lowest-effort highest-credibility upgrade in the whole roadmap. **Effort: quick-win.** *First step:* add a `wilson(k, n)` helper in `src/bible_eval/cli.py`, store `{value, lo, hi, n}` on each summary metric, draw whiskers in `docs/app.js` `svgBarChart`.
- **Noise Bands** — *Bootstrap CIs for continuous metrics (WER/CER/fuzzy/chatter).* Resample the per-verse list 1000× → 2.5/97.5 percentiles; render faint bands behind the trend line. **Effort: quick-win.** *First step:* `cli.py` already has the per-verse `results` array in memory at summary time — bootstrap there.
- **Repro Stamp** — *lm-eval-harness-style `repro` block on every run.* git commit, scorer/normalizer version, seed, decode params, quant, prompt-mode hash, plus a one-click "Copy reproduce command" button. **Effort: quick-win.** *First step:* extend the summary dict in `cli.py.cmd_run` and surface it on the "Latest Run" card in `docs/app.js`.

### Scoring correctness (audit-driven, will move numbers)
- **No-False-Fails Normalization** — *Strip leading verse-references/digits, normalize em/en dashes and Unicode NFC, add a punctuation-insensitive scoring tier.* Today "In the beginning God created the heaven and the earth." vs "...the heavens and the earth" scores `wer=0.300 / inaccurate_recall` for what is essentially correct, and `John 3:16` prepended by the model inflates WER. **Effort: quick-win.** *First step:* extend `NormalizationConfig` in `src/bible_eval/core/normalizer.py`, mirror the existing `strip_thinking` postprocess-flag pattern; pin cases in `tests/test_normalizer.py`.
- **Verbatim Coverage** — *Longest contiguous common token span + count of exact n-grams (n≥5) in `Scorer.score_pair`.* A 28-word verbatim block + one paraphrased clause is qualitatively different from 28 scattered correct words; WER can't tell them apart, and `token_sort_ratio` is order-blind (a scrambled verse scores high) — a real hole for *verbatim* scoring. **Effort: quick-win.** *First step:* `core/scorer.py` already imports `difflib.SequenceMatcher` — emit `lcs_token_ratio` and use it to harden the 4-way label.

### Front-end (pure JS over data we already emit)
- **Hash Router** — *Deep-linkable URL state for every view* (`#run=…&model=…&verse=John+3:16&view=diff`). ~30 lines, no library. The connective tissue that makes every other view shareable. **Effort: quick-win.** *First step:* parse/restore `location.hash` against the existing `<select>` values in `docs/app.js`.
- **Graded Diff Heatmap** — *Upgrade `renderDiff` from binary del/ins to a continuous severity gradient* (substitution = hot, near-miss like "thy"/"thine" = light). The whole thesis is verbatim fidelity, yet the diff treats a one-letter slip and a total swap identically. **Effort: quick-win.** *First step:* `diffOps` already aligns tokens — interpolate background-color per op.
- **Hall of Shame** — *A shareable wall of the funniest fluent-but-wrong hallucinations* (`label=='total_hallucination'` with high `token_sort_ratio`), with copy/share buttons and an optional curated `docs/data/hall_of_shame.json`. Viral and pedagogically sharp. **Effort: quick-win.**
- **Error-Composition Bars** — *Stacked bars of substitutions / deletions / insertions per model.* Insertions/chatter = compliance failure; substitutions/deletions = knowledge failure — the K-vs-C story, decomposed. Fields already exist. **Effort: quick-win.**

### Cheap, high-impact experiments & content
- **Abstention Probes** — described in §1.3. **Effort: quick-win.** *First step:* add void-reference item types to `engine/sampler.py`, refusal detection + an `abstention_rate` / `fabricated_on_void` label in `core/scorer.py`.
- **Confidence Suffix** — *Optional prompt suffix asking for a 0–100 confidence, parsed pre-scoring; report ECE/Brier/overconfidence-rate.* A model that confidently emits a wrong verse is the named harm in the religious-LLM literature. **Effort: quick-win.** *First step:* prompt template in `config.yaml` + a parser/binner in `cli.py`.
- **Translation-Bleed Probe** — *Ask for a named translation, score against THAT translation, flag when output matches a different one better* (asked ASV, got KJV). Doubles as a contamination signal for which translation dominates training. **Effort: quick-win** *(once WEB/ASV data lands — see §5).* *First step:* exploit the existing `verse_id` alignment; `Scorer` is reused unchanged.
- **Quant-Fidelity Curve** — *Run one model across F16/Q8/Q6/Q5/Q4 and plot exact-verbatim rate vs quant.* "Which quant can still quote a verse word-for-word?" is an open empirical question for the entire Ollama crowd, and verbatim exact-match is exactly the brittle capability MMLU hides. Near-zero new code. **Effort: quick-win.**

### Engineering hygiene (see §5 for the full list)
- **CI workflow** (ruff + pytest) and **dependency cleanup** (declare `rapidfuzz`, drop unused `tqdm`) are both quick-wins and should ride along this week.

---

## 3. Medium features

### The thesis, visualized
- **Knowledge-vs-Compliance Quadrant** — described in §1.2. **Effort: medium.** *First step:* new SVG scatter in `docs/app.js` reusing `svgLineChart`'s axis/scale code; quadrant guide-lines + labels ("Verbatim machine", "Knows-but-chatty", "Confident-but-wrong", "Refuses").
- **Verse × Model Heatmap** — described in §1.4. **Effort: medium.** *First step:* SVG `<rect>` grid in `docs/app.js`; click a cell → existing `renderDiff`; metric toggle (label / WER / CER) reusing the `<select>` pattern.
- **HELM-Style Category Grid + Memorization Gradient** — *Per-segment aggregation (Testament, book genre, length bucket, famous/obscure tier) → models × categories heatmap; report `acc(famous) − acc(obscure)`.* Turns the benchmark into a popularity-vs-knowledge diagnostic. **Effort: medium.** *First step:* add a `breakdowns` object in `cli.py`; seed a popularity tier in `data/taxonomy.json`.

### New task types (the literature's "where do they fail")
- **Prefix-Completion / Extraction Threshold** — *Give the first K% of a verse and score only the held-out suffix; sweep K∈{25,50,75}% and record the minimum prefix at which it's reproduced verbatim.* The canonical discoverable-extraction probe; cleanly separates priming-recall from cold recall. **Effort: medium.** *First step:* new `prompt_mode` in the dispatch in `engine/interrogator.py._build` (~lines 76–88); reuse `Scorer` on `(suffix_gt, suffix_pred)`. The `verse_id` ordering also enables a zero-leakage "prior verse → next verse" continuation.
- **Find-the-Misquote** — *Deterministically corrupt a verse to a target WER bucket and ask the model to classify verbatim/altered or return the fix.* Probes verification, not recall; test-case generation is free because we own the ground truth and the edit machinery. **Effort: medium.**
- **DCQ Contamination Quiz** — *Multiple-choice "spot the canonical verse" among word-perturbed decoys, per translation.* Black-box "has the model SEEN this?" signal needing no logprobs; the per-translation contamination map is genuinely novel. **Effort: medium.**
- **Query-by-Description Retrieval** — *Paraphrased description → model returns reference + verbatim text; score reference exact-match separately from text WER.* Kaiser found this is the *strongest* capability across models, so it's a great calibration baseline against weak continuation. **Effort: medium.**
- **Cross-Version Blend Detection** — *Score a prediction against ALL loaded translations and flag when the closest ≠ the asked one.* Blending is the single most-cited failure mode in the only comparable prior work (benkaiser.dev) and nobody has tooled it. **Effort: medium.** *First step:* requires WEB/ASV data (§5); `Scorer` reused via `verse_id` alignment; emit `best_match_version` / `blend_flag`.

### Metrics depth
- **Semantic Floor** — *Optional embedding similarity (sentence-transformers, lazy-imported; rapidfuzz partial-ratio offline fallback) → 2D map of surface fidelity vs meaning.* High-meaning/low-surface = faithful paraphrase ("knows it, won't quote it"); low-meaning = true hallucination. Directly disentangles the K-vs-C axis. **Effort: medium.** *Keep heavy deps optional to preserve the dependency-light ethos.*
- **Paired McNemar Comparison** — *`bible-eval compare --a modelA --b modelB` → 2×2 discordant table → exact-binomial McNemar p-value; head-to-head matrix with significance badges on the site.* The statistically correct way to ask "is A actually better than B?" on a shared verse set. **Effort: medium.**
- **AURC / Selective Prediction** — *Add an "I DON'T KNOW" affordance and score with a risk-coverage curve.* Rewards graceful abstention over fabrication — the deployment-facing metric. **Effort: medium.** *Pairs with the confidence suffix and abstention probes.*
- **Prompt-Perturbation Robustness** — *Re-query each verse under 3–5 meaning-preserving template variants; report mean accuracy, spread, and verdict-consistency.* A single perturbation flips rankings ~63% of the time, so single-template numbers are fragile. Generalizes the existing naive/constraint/system2 design into a controlled grid. **Effort: medium.**

### Platform & ops
- **Latency/Token/Cost Capture → Pareto view** — *Connectors return `(text, usage)`; persist `latency_ms`/tokens per verse; scatter accuracy vs latency-or-model-size with a Pareto frontier.* For local models the real budget is VRAM and seconds — "what's the smallest/fastest model that still quotes scripture?" **Effort: medium.** *First step:* read the discarded `eval_count`/`usage` fields in `connectors/ollama.py` and `connectors/openai_compatible.py`; wrap calls with `time.perf_counter` in `engine/interrogator.py`.
- **Regression Diff View** — *`bible-eval diff --base RUN_A --head RUN_B` + a tri-pane site mode (regressed / fixed / unchanged), with deltas gated by the bootstrap CI so you don't chase noise.* Answers "did upgrading the model or switching naive→system2 actually help?" **Effort: medium.**
- **Faceted Sample Browser** — *Generalize "Examples" into a sortable/filterable table (ref/label/WER/CER/chatter), free-text search, click-to-diff, and "order by verse across models."* Inspect's killer feature; the cross-model pivot exposes universally hard verses. Pure front-end. **Effort: medium.**
- **Real RAG Regime** — *A tiny local BM25/embedding retriever over the verse JSON injects candidate verses, scored against parametric (naive/constraint) and the fake-DB system2.* Surprising built-in angle: RAG can *hurt* exact-match by retrieving a wrong-translation verse. **Effort: medium.**
- **Frozen Canary Set + Task Hash** — *Pin a named verse set, stamp every run with `canary_set_version` + content hash + scorer version; the trend line only connects runs sharing the hash (else a visible warning).* Prevents the classic "trend across secretly-different verses" lie. **Effort: medium.** *First step:* add to `results/store.py` summary schema + a guard in `docs/app.js`.
- **Self-Maintaining CI Bot** — *Scheduled GitHub Action (self-hosted Ollama runner) pulls new model tags, runs the canary set, appends to `history.json`, regenerates the site, opens a PR with the new rank.* Makes the leaderboard a living artifact. **Effort: medium.**
- **HF Dataset Release** — *Package per-run JSON + canary set as a versioned Hugging Face dataset + a "detect+correct" task variant mirroring IslamicEval subtask 1.* Lowest-effort path to being discovered and cited. **Effort: medium.**
- **Multi-Model Trend Overlay** — *All selected models on one trend chart with toggles + a dashed reference-baseline ceiling, plus prompt-regime small-multiples and a cheap `stroke-dashoffset` draw-in animation.* **Effort: medium.**
- **Side-by-Side Model Diff** + **Daily Verse Showdown** (scheduled "verse of the day" card + RSS feed; recurring content engine and a naturally growing time series). **Effort: medium / quick-win respectively.**

---

## 4. Ambitious / moonshot

- **Logprob Memorization Suite (Min-K% / perplexity / ReCaLL)** — *Capability-gated on the OpenAI-compatible connector (vLLM/llama.cpp expose logprobs): score the model's probability mass on the ground-truth verse, not just its generation.* Cross-validates behavioral labels with a probabilistic signal — the question no WER-based eval can answer: *memorized or reconstructed?* **Effort: ambitious.** *Ollama degrades gracefully via a "requires logprobs" flag.*
- **Multi-Epoch Self-Consistency** — *`epochs: N` config; query each verse N× at temp>0; report per-verse consistency and flag "flaky" verses.* "Verbatim 1/5 times" is a very different finding from 5/5 — strengthens knowledge-vs-luck. **Effort: ambitious.** *Touches the run pipeline in `cli.py.cmd_run`.*
- **Multilingual Recall via eBible/OPUS** — *Ingest verse-aligned PD translations (Reina-Valera, Luther 1912, Vulgate); report per-language WER and the English-vs-non-English gap.* Free, already verse-aligned to our `book/chapter/verse` keys, and the literature shows every model degrades sharply outside English. **Effort: ambitious.**
- **Generalize to Any Canonical PD Text** — *Pluggable "corpus" abstraction: US Constitution articles, Shakespeare act/scene/line, Gutenberg stanzas.* The scoring core is already text-agnostic; only the loader and prompt are Bible-specific. Multiplies the audience ("can Llama recite the Second Amendment?") with no engine change. **Effort: ambitious** *(mostly loader + prompt).*
- **Deuterocanon Long-Tail Axis** — *Add a deuterocanonical canon variant (1611 KJV Apocrypha, Brenton LXX) and compare verbatim rate vs the 66-book canon* — a clean training-data-frequency cliff. **Effort: medium-ambitious.**
- **LLM-as-Judge Compliance Taxonomy** — *Judge classifies non-verse content (preamble / trailing exegesis / citation wrapper / refusal), but rules stay primary (IFEval style) and we report judge-vs-rule Cohen's κ + judge self-consistency.* A real compliance taxonomy, honest about judge reliability. **Effort: ambitious.**
- **"Stump the Model" Arena** — *LMArena-style blind pairwise verse-recall battles with verbatim-fidelity Elo + community-submitted stumper verses (pre-generated offline to keep the site static).* The proven viral mechanic, aimed at a charged domain, growing the hard-set dataset for free. **Effort: ambitious.**
- **Rich Model Cards w/ Prompt-Regime Sweep + Auto-Markdown Export** — *Per-model permalink pages (CIs, quadrant position, naive/constraint/system2 strip showing what system2 actually buys) + a copy-paste Markdown block for a model's README.* The regime sweep on one card directly demonstrates the compliance half of the thesis. **Effort: ambitious.**

---

## 5. Engineering-health improvements (from the audit)

Sequencing note: do the **CI workflow, dependency fix, and data validation** before any statistics or scoring changes — they protect every subsequent edit. The normalization/coverage scorer changes pair naturally with the CIs so you can *see* whether a change is statistically real.

### Quick wins
- **CI workflow** — no `.github/workflows` exists; pytest/ruff never run automatically and `docs/` is hand-committed. *Add a workflow running `ruff check`, `ruff format --check`, `pytest` on Python 3.9–3.12, plus an optional job that runs `bible-eval export-site` and publishes Pages so the leaderboard can't drift from `results/history.json`.* **quick-win.**
- **Dependency truth** — `tqdm` is declared but never imported; `rapidfuzz` **is** used in `core/scorer.py` but **undeclared**, so `token_sort_ratio` silently differs across environments (rapidfuzz vs difflib fallback). *Drop tqdm (or actually wrap the verse loop), declare+pin rapidfuzz (or a `[fuzzy]` extra), consolidate metadata into `pyproject.toml [project]`.* **quick-win.**
- **Data validation** — `VerseDatabase.from_raw_json` (`data/loader.py`) silently overwrites duplicate ids, doesn't check empty text, and raises a mid-run `KeyError` on unknown books ("Psalm" vs "Psalms"). *Validate at load (duplicate ids, empty text, unknown books) and fail fast; add a `bible-eval validate-data --config` subcommand* so WEB/ASV contributors get instant feedback. **quick-win.**
- **Connector error handling** — `ollama.py`/`openai_compatible.py` call `urlopen` with no try/except, so a 404 (model not pulled) becomes a vague "failed after retries" and the interrogator retries blindly. *Wrap urlopen → typed `ConnectorError` with status+body; distinguish non-retryable 4xx from retryable 5xx/timeouts.* **quick-win.**

### Medium
- **Hosted-API connectors + max_tokens** — there are NO hosted connectors (only ollama/openai_compatible/reference), and `openai_compatible.py` never sets `max_tokens`, so local servers over-generate. *Add thin urllib Anthropic (Messages API) and Gemini (generateContent) connectors registered in `interrogator.from_model_config`; add `max_tokens`/`num_predict` to `OpenAICompatibleOptions`.* **medium.**
- **Extract testable aggregation** — grading/labeling/notes are a ~100-line inline block in `cli.py.cmd_run` (~121–217), untested. Worse, **`clean_output_rate` is derived only from `strip_thinking_changed`**, so a model that chatters without a `<think>` block reports a misleadingly perfect clean rate. *Extract `summarize_results(results) -> RunSummary` into `results/aggregate.py`; redefine `clean_output_rate` from the actual `verbatim` label fraction.* **medium.**
- **Response cache** — `cmd_run` re-queries every model×verse on each run; a scorer fix re-pays full inference. *Content-addressed cache `results/cache/<sha256(model+options+prompt)>.json` consulted in `interrogator._generate`, with `--no-cache`.* Makes scorer/normalizer iteration free. **medium.**
- **Path handling** — `cmd_run` is hard-coded to CWD-relative paths, writes the summary **twice** (throwaway `runs/<id>` + durable `results/runs`), and stores repo-relative paths that break `examples`/`export-site` when run elsewhere. *Add `--out/--results-dir/--docs-dir`, resolve from a project root, drop the duplicate write.* **medium.**
- **Test coverage** — only 5 tests; zero for the sampler (determinism/stratification), loader (alias/dup/empty), `_strip_thinking`, connectors (mocked HTTP), or `store.append_run`. *Add fast pure-logic tests for each before any refactor.* **medium.**
- **Site schema versioning** — `history.json` has no `schema_version`; newer runs rely on fields (`avg_len_ratio`, `clean_output_rate`) older committed runs lack, with no migration. *Add `schema_version` in `results/store.py`, guard missing fields in `app.js` (extend the existing `Number.isFinite` pattern), add prompt_mode/version UI filters.* **medium.**

### Ambitious
- **Real corpora** — only a 10-verse KJV sample across 5 books ships; WEB/ASV raw files don't exist, so advertised multi-translation support isn't actually shippable and the stratified sampler can't meaningfully stratify. *Ship PD KJV/WEB/ASV via a `bible-eval fetch-data` command/downloader + a committed canonical 100–200 verse stratified canary fixture.* **ambitious — and a hard dependency for cross-version blend detection, translation-bleed, and meaningful stratification.** This is the single most enabling piece of infrastructure on the list; prioritize the WEB/ASV fetch even ahead of some quick wins if you want the §3 cross-version features.

---

### Recommended first sprint (concrete)
1. CI workflow + dependency fix + data validation (protect the codebase).
2. Wilson + bootstrap CIs and the repro stamp (credibility).
3. No-false-fails normalization + verbatim-coverage metric, with the CIs proving the change is real.
4. Hash router + Knowledge-vs-Compliance quadrant + Verse×Model heatmap (the thesis, finally visible).
5. Abstention probes + `fetch-data` for WEB/ASV (unlock the cross-version and safety angles).

Files that recur as insertion points: `src/bible_eval/cli.py` (summary schema, aggregation), `src/bible_eval/core/scorer.py` + `normalizer.py` (metrics), `src/bible_eval/engine/interrogator.py` (`_build` prompt dispatch ~76–88, timing), `src/bible_eval/engine/sampler.py` (probes/tiers), `src/bible_eval/data/loader.py` + `data/taxonomy.json` (validation, popularity, canon), and `docs/app.js` (all visualizations, hand-rolled SVG + `diffOps` + `renderDiff`).