# Study Design: A Verbatim Scripture-Recall Benchmark for Open-Weight LLMs

*This document defines the methodology that makes `bible-eval` a rigorous, defensible
study rather than a casual leaderboard. It is grounded in a fact-checked literature
review (see [Evidence & citations](#evidence--citations)). Recommendations are tagged
**[verified]** (supported by a cited primary source) or **[provisional]** (our design
choice, not yet externally validated).*

## 1. Thesis

`bible-eval` measures **verbatim fidelity** when open-weight LLMs quote public-domain
scripture (KJV/WEB/ASV), separating two axes:

- **Knowledge** — does the model know the verse? (content accuracy, WER/CER, fuzzy similarity)
- **Compliance** — can it output *only* the verse, no chatter? (clean-output rate, chatter ratio)

The scientific hook is that **verbatim recall *is* the memorization phenomenon**: scripture
is among the most-duplicated public-domain text on the web, so this benchmark is a clean,
license-safe probe of training-data memorization across model families, sizes, and
quantization tiers.

## 2. Core design principles

### 2.1 Deployment-aware, multi-objective — not a single score **[verified]**

Frame results as an **operating-point problem**, not one number. Report accuracy
*alongside* latency, peak VRAM, and prompt-sensitivity, and show the **Pareto front** —
the highest-scoring config is frequently not the best practical deployment point
(arXiv:2604.07035). Use a **fully balanced design**: every `model × prompt-strategy ×
text-condition` cell evaluated on the *same* item subset, each reported with Wilson CIs.

*Status in repo:* latency ✅, confidence intervals ✅, three prompt regimes ✅,
Knowledge-vs-Compliance quadrant ✅. **To add:** peak-VRAM capture, an
accuracy-vs-cost Pareto view, and enforcing the balanced grid in `cmd_run`.

### 2.2 Confidence intervals + efficient sampling **[verified]**

Every rate carries a **Wilson** interval; every continuous metric a **bootstrap**
interval (both shipped). To scale beyond a 10-verse smoke test toward the full Bible
without a linear cost blow-up, adopt statistically-efficient adaptive sampling:
**Factorized Active Querying** matches a target CI width with up to ~5× fewer queries
under valid frequentist coverage (arXiv:2601.20251). Treat the verse set as
finite-population inference.

### 2.3 Prompt-perturbation robustness is mandatory **[verified]**

Single-prompt rankings are not trustworthy: a single semantics-preserving perturbation
reorders model rankings in **63%** of cases (arXiv:2603.13285), and format changes alone
move models up to 8 leaderboard positions (Kendall's τ ≈ 0.46; arXiv:2402.01781).
**Requirement:** query each verse under several meaning-preserving prompt variants
(e.g. *"Quote John 3:16"*, *"What does John 3:16 say?"*, with/without translation named)
and **report ranking stability**, not just a point score.

### 2.4 Verbatim recall via probabilistic discoverable extraction **[verified]**

Prefer **probabilistic extraction** over a single greedy decode: prompt with an
*a*-token prefix and estimate the probability of the exact *j*-token suffix, then report
**extraction coverage** — the fraction of a text inside a suffix whose extraction
probability exceeds a threshold τ (arXiv:2505.12546; Hayes et al., NAACL 2025). Greedy
decoding ignores sampling non-determinism and under-reports memorization.

*Engineering note:* this needs token-level logprobs. Ollama exposes these only partially,
so the practical port is a **prefix-completion task** (give the first K% of a verse, score
the held-out suffix) with an *n*-sample temperature estimate as the fallback when full
probabilities are unavailable.

### 2.5 Contamination handling — reframed for a memorization benchmark **[verified]**

In ordinary benchmarks, train/test overlap *inflates* scores and can invalidate
conclusions; detection splits into white/gray/black-box methods, and **CDD** can flag
contamination from sampled outputs alone (+21.8–30.2% over baselines) while **TED**
corrects inflated scores (up to 66.9% mitigation) (arXiv:2402.15938; survey
arXiv:2502.14425).

**The twist for us:** here memorization is the *desired capability*, not a confound. So we
don't try to remove it — we *characterize* it. The headline analysis is the
**memorization gradient**: accuracy as a function of verse popularity (famous → obscure)
and model capacity. Memorization is known to grow log-linearly with model capacity,
training-duplication, and prompt-context length, and can occur even from a single training
document (arXiv:2202.07646, ICLR 2023; arXiv:2012.07805, USENIX 2021). This motivates
three explicit axes:

1. **Capacity** — compare across model sizes within a family.
2. **Duplication** — stratify verses by popularity / web-frequency.
3. **Context** — vary prompt framing (bare reference vs book/chapter context).

> **First result (Gemma 4, full KJV, N=60):** famous verses **30%** verbatim
> [17–48%] vs obscure verses **0%** [0–11%] — a **+30-point gap with
> non-overlapping CIs**. The model recalls only the most-duplicated verses and
> *zero* obscure ones: recall tracks popularity, not knowledge. This is the
> memorization signature the framing predicts.

### 2.6 Avoid naive Elo / arena leaderboards **[verified]**

Elo is sensitive to match order and hyperparameters, transitivity is not guaranteed, and
instability is worst for closely-matched models (arXiv:2311.17295). **Prefer absolute
metrics with CIs** (WER/CER, strict accuracy, extraction coverage). If pairwise comparison
is wanted, fit **Bradley–Terry with bootstrap intervals** (the model LMSYS moved to) and
accept that statistically-tied models are tied.

### 2.7 LLM-as-judge: use with caution **[verified — as a caution]**

LLM-as-judge is one option for scoring chatter/compliance, but the literature did not
confirm which judge biases dominate (two strong bias claims were *refuted* under
adversarial verification). Keep **rule-based scoring primary** (as today); if a judge is
added, report judge-vs-rule agreement (Cohen's κ) and judge self-consistency rather than
trusting it outright.

## 3. Model roster

### 3.1 Report an openness tier, not just a name **[verified]**

Classify every model on Ai2's axis: **open-weight** (weights only) vs **partially-open**
vs **fully-open-science** (weights + training data + code + recipes + checkpoints)
(allenai.org/blog/olmo2). Report this tier *and* the license tier (genuinely open vs
acceptable-use-restricted) as columns, because the openness tier determines what claims
the study can defend.

### 3.2 Anchor with fully-open-data models for ground-truth contamination **[verified]**

Only **fully-open-data** models let you *verify* whether a given verse string was in
training data — closed-data models cannot support this. Include **OLMo 2** (7B/13B, and
32B from March 2025) and **EleutherAI Pythia** (trained on the public Pile, identical data
order, 154 checkpoints) as anchors (allenai.org/blog/olmo2; github.com/EleutherAI/pythia).
Pythia further ships ready-built memorization tooling (Biderman et al., *Emergent and
Predictable Memorization*, NeurIPS 2023, arXiv:2304.11158). *(Note: OLMo is not unique
here — Pythia is an equal or stronger option; cite both.)*

### 3.3 Membership inference: Min-K% Prob **[verified]**

To estimate "did this model see this verse," adopt/cite **Min-K% Prob** — reference-free
(no training-corpus knowledge, no extra training), averaging the log-likelihood of the
*k%* lowest-probability tokens (best *k*=20), ~0.72 average AUC on WikiMIA
(arXiv:2310.16789, ICLR 2024). Treat the signal as **probabilistic, not binary**, and
corroborate against Pile/OLMo ground truth where possible.

### 3.4 Report per-text, never just aggregate **[verified]**

Memorization is highly model- and text-specific: Llama 3.1 70B memorizes *entire* books
(90.89% extraction coverage of *Harry Potter* at τ=1%) while averaging only ~0.6% across
random Books3 sequences — the mean hides near-complete memorization of specific works
(arXiv:2505.12546). **Mandate:** report per-verse / per-book results (the verse×model
heatmap and popularity gradient), not a single aggregate.

### 3.5 Quantization tiers **[verified]**

Quantization degrades recall **modestly and non-monotonically** — lower-bit models don't
uniformly underperform — but the effect is **amplified in smaller models**, so scrutinize
sub-8B models most (arXiv:2505.13963). **4-bit (Q4_K_M) is the practical floor**: 4-bit
preserves emergent abilities while 2-bit (and naive RTN 3-bit) collapse
(arXiv:2307.08072; arXiv:2404.14047). So **FP16 / Q8 / Q5_K_M / Q4_K_M are all in the safe
regime** for the sweep; do not claim a precise Q4-vs-FP16 loss figure (the "~2% loss"
claim failed verification). The question *"which quant can still quote scripture
verbatim?"* is open and citable-worthy — near-zero extra code.

### 3.6 Recommended roster **[provisional]**

Families × a **size ladder** (capacity axis of the memorization gradient) × **quant
tiers**: Llama 3.x, Qwen 2.5/3, Gemma 2/3/4, Mistral/Mixtral, DeepSeek R1/V3, Phi-4, plus
the **OLMo 2 / Pythia** open-data anchors; optionally Falcon, Yi, Command-R. Refresh at run
time and record each model's exact license + openness tier.

## 4. Positioning vs related work

### 4.1 Closest analogue: IslamicEval 2025 **[verified]**

The nearest prior art is **IslamicEval 2025** (ArabicNLP @ EMNLP) — the *first shared task*
on detecting/correcting LLM hallucination in quoted scripture (Quran Ayahs and Hadiths)
plus scripture QA (aclanthology.org/2025.arabicnlp-sharedtasks.67). Two things transfer
directly:

1. **Evaluation design** — strict exact-match with *superficial* normalization (they strip
   diacritics; we already collapse whitespace/quotes), justified because minor deviations
   change scriptural meaning. This is exactly bible-eval's `verbatim` label philosophy.
2. **The premise is validated** — open-model fidelity gaps are large and discriminating:
   Arabic-tuned ALLaM-7B scored **84.06%** correct Quran verses vs **4.77%** for
   Llama-3.1-8B-Instruct and **6.86%** for Qwen3-8B, and all models did far worse on Hadith
   than Quran. Domain/language tuning dominated raw size — a result worth replicating for
   English Bibles.

### 4.2 Contrast references (not direct comparators) **[verified]**

Bible **translation/alignment** corpora measure BLEU-scored NMT, not LLM verbatim recall:
**eBible** (1009 translations, 833 languages; arXiv:2304.09919) and the curated
**BibleNLP/ebible** (verse-only, `vref`-aligned parallel text;
github.com/BibleNLP/ebible). They are positioning contrasts — and a useful **multilingual
reference-text source** for a future cross-language recall extension — but not the same
task.

### 4.3 Differentiation **[verified gap]**

No Bible-specific verbatim-**recall** LLM benchmark surfaced in the verified literature
(only an informal blog exploration, benkaiser.dev). bible-eval's novelty:

- **First verbatim-recall** (not QA, not translation) evaluation on **public-domain English
  Bibles** — license-safe by construction.
- **Memorization-framed**, with per-text reporting and open-data anchors to separate
  memorization from reconstruction.
- Position as the **open Bible counterpart to IslamicEval**.

## 5. Reproducibility checklist

- [x] Per-run **reproduce stamp** (git commit, seed, prompt hash, tool/python version).
- [x] Deterministic sampling (seeded) and seeded bootstrap.
- [x] Immutable public-domain ground truth; versioned `schema_version`.
- [ ] Pin a named, content-hashed **canary verse set**; only connect trend points sharing
      the hash.
- [ ] Capture decode params + quant + peak VRAM per model in the summary.
- [ ] Publish the dataset + per-run JSON (e.g. a versioned Hugging Face dataset release).

## 6. Implementation roadmap (derived from this design)

| Priority | Item | Principle | Effort |
|---|---|---|---|
| 1 | Prompt-perturbation harness + ranking-stability report | §2.3 | medium |
| 2 | Quantization-fidelity sweep (FP16→Q4 on one model) | §2.5 / §3 | quick |
| 3 | Popularity-stratified memorization gradient | §2.5 | medium |
| 4 | Prefix-completion / extraction-coverage task | §2.4 | medium |
| 5 | Peak-VRAM capture + accuracy-vs-cost Pareto view | §2.1 | medium |
| 6 | Efficient adaptive sampling (FAQ) | §2.2 | ambitious |

See `docs/ROADMAP.md` for the broader feature backlog.

## Evidence & citations

All claims tagged **[verified]** trace to primary sources confirmed by 3-vote adversarial
verification:

- **Deployment-aware multi-objective design** — arXiv:2604.07035
- **Probabilistic discoverable extraction / extraction coverage** — arXiv:2505.12546; Hayes et al., NAACL 2025
- **Contamination detection/correction (CDD/TED)** — arXiv:2402.15938 (ACL 2024 Findings); survey arXiv:2502.14425
- **Memorization scaling laws** — arXiv:2202.07646 (ICLR 2023); arXiv:2012.07805 (USENIX 2021)
- **Prompt-perturbation instability** — arXiv:2603.13285; arXiv:2402.01781 (ACL 2024); arXiv:2405.19740
- **Efficient finite-population sampling (FAQ)** — arXiv:2601.20251
- **Elo instability** — arXiv:2311.17295
- **Openness taxonomy / fully-open-data models** — allenai.org/blog/olmo2; github.com/EleutherAI/pythia
- **Predictable memorization tooling** — Biderman et al., NeurIPS 2023, arXiv:2304.11158
- **Membership inference (Min-K% Prob)** — arXiv:2310.16789 (ICLR 2024)
- **Per-text memorization specificity** — arXiv:2505.12546
- **Quantization effects on recall** — arXiv:2505.13963; arXiv:2307.08072 (EMNLP Findings 2023); arXiv:2404.14047
- **IslamicEval 2025 (closest analogue)** — aclanthology.org/2025.arabicnlp-sharedtasks.67
- **Bible translation corpora (contrast refs)** — eBible arXiv:2304.09919; github.com/BibleNLP/ebible

*Refuted under verification (do not cite): a dominant "style bias" in LLM-as-judge; a
universal judge brevity bias; "perturbations explain ~half of variance"; that OLMo 2
*uniquely* enables contamination analysis (Pythia does too); that Q4 quantization recovers
unlearned knowledge; a precise "~2% Q4-vs-FP16 loss" figure.*
