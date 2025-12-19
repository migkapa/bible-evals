# Metrics

## WER (Word Error Rate)

`WER = (S + D + I) / N`

- `S` substitutions, `D` deletions, `I` insertions (word-level)
- `N` reference word count

## CER (Character Error Rate)

Character-level edit distance divided by reference character count.

## Chatter ratio

Measures length drift vs ground truth:

`CR = (len(pred) - len(gt)) / len(gt)`

Computed on *raw strings with whitespace collapsed* so “Here is the verse…” is penalized.

## Fuzzy similarity

Token-sort similarity on normalized text to separate “paraphrase” from “total miss”.

