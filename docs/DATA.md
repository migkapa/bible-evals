# Data

## Licensing

This repository is designed to work with **public domain** Bible texts (KJV 1769, WEB, ASV).
Do not commit copyrighted translations (NIV/ESV/etc.) into a public repo.

## Input schema (raw)

`bible-eval` expects a JSON array of objects with:

- `book` (string): book name (e.g. `Genesis`, `1 John`, `Psalm`).
- `chapter` (int)
- `verse` (int)
- `text` (string)

Example:

```json
{"book":"John","chapter":3,"verse":16,"text":"For God so loved..."}
```

## Canonical IDs

We map verses to a deterministic integer:

`ID = (book_index * 1_000_000) + (chapter * 1_000) + verse`

This makes sorting and lookups fast and stable across files.

## Normalization (for scoring)

Scoring uses a normalization pipeline to avoid penalizing typographic artifacts:

- remove `[` `]` and `¶`
- normalize smart quotes to straight quotes
- normalize whitespace (collapse runs)

“Strict mode” still penalizes added/omitted words and punctuation drift after normalization.

