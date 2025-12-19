# Architecture (Project Genesis)

## Goals

- Local, immutable **ground truth** datasets (public domain).
- Multiple **prompting regimes** to separate recall vs compliance.
- Metrics that quantify **verbatim fidelity** (WER/CER) plus **chatter**.
- Connector abstraction so you can evaluate **open-source local models**.

## Package layout

- `src/bible_eval/data/`: load/validate verse datasets and taxonomy.
- `src/bible_eval/utils/`: canonical verse ID encoding and reference parsing.
- `src/bible_eval/core/`: normalization + scoring metrics.
- `src/bible_eval/connectors/`: model adapters (Ollama, OpenAI-compatible).
- `src/bible_eval/engine/`: sampling + interrogation loop.

## Data flow

1) Load taxonomy (`data/taxonomy.json`) to map book names/aliases → `book_index`.
2) Load raw verse JSON → `VerseRecord`s keyed by canonical ID.
3) Sample a canary set (random or stratified).
4) Prompt a model and capture raw output.
5) Score against ground truth:
   - strict (exact after normalization),
   - WER/CER,
   - chatter ratio,
   - fuzzy similarity.
6) Persist results to `runs/<timestamp>/results.json`.

