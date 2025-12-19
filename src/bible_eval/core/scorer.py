from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Optional

from bible_eval.core.normalizer import NormalizationConfig, Normalizer


def _levenshtein(a: list[str], b: list[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # DP with two rows.
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        cur = [i]
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _word_edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """
    Return (distance, substitutions, deletions, insertions) at word level.

    Distance equals S + D + I.
    """
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # match/substitution
            )

    i, j = n, m
    substitutions = deletions = insertions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 1:
                    substitutions += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
            continue
        # Should be unreachable; fall back defensively.
        if i > 0 and j > 0:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    distance = substitutions + deletions + insertions
    return distance, substitutions, deletions, insertions


def _try_rapidfuzz_ratio(a: str, b: str) -> Optional[float]:
    try:
        from rapidfuzz.fuzz import ratio  # type: ignore
    except Exception:
        return None
    return float(ratio(a, b))


def _token_sort_ratio(a: str, b: str) -> float:
    def toks(s: str) -> list[str]:
        return [t for t in s.split(" ") if t]

    sa = " ".join(sorted(toks(a)))
    sb = " ".join(sorted(toks(b)))
    rf = _try_rapidfuzz_ratio(sa, sb)
    if rf is not None:
        return rf
    return 100.0 * SequenceMatcher(None, sa, sb).ratio()


@dataclass(frozen=True)
class ScoreConfig:
    normalization: NormalizationConfig = NormalizationConfig()
    fuzzy_casefold: bool = True


@dataclass(frozen=True)
class ScoreResult:
    wer: float
    cer: float
    chatter_ratio: float
    token_sort_ratio: float
    substitutions: int
    deletions: int
    insertions: int
    ref_words: int
    gt_norm: str
    pred_norm: str
    label: str
    contains_gt: bool


class Scorer:
    def __init__(self, cfg: Optional[ScoreConfig] = None) -> None:
        self.cfg = cfg or ScoreConfig()
        self._norm = Normalizer(self.cfg.normalization)
        base = asdict(self.cfg.normalization)
        base["casefold"] = self.cfg.fuzzy_casefold
        self._norm_fuzzy = Normalizer(NormalizationConfig(**base))

    def score_pair(self, gt: str, pred: str) -> ScoreResult:
        gt_norm = self._norm.normalize(gt)
        pred_norm = self._norm.normalize(pred)

        gt_words = gt_norm.split(" ") if gt_norm else []
        pred_words = pred_norm.split(" ") if pred_norm else []

        distance, substitutions, deletions, insertions = _word_edit_counts(gt_words, pred_words)
        wer = (distance / len(gt_words)) if gt_words else (0.0 if not pred_words else 1.0)

        gt_chars = list(gt_norm)
        pred_chars = list(pred_norm)
        char_edits = _levenshtein(gt_chars, pred_chars)
        cer = (char_edits / len(gt_chars)) if gt_chars else (0.0 if not pred_chars else 1.0)

        # Chatter is computed on raw(ish) text: collapse whitespace, keep casing and punctuation.
        raw_norm = Normalizer(NormalizationConfig(casefold=False)).normalize
        gt_raw = raw_norm(gt)
        pred_raw = raw_norm(pred)
        chatter_ratio = ((len(pred_raw) - len(gt_raw)) / len(gt_raw)) if gt_raw else (0.0 if not pred_raw else 1.0)

        gt_fuzzy = self._norm_fuzzy.normalize(gt)
        pred_fuzzy = self._norm_fuzzy.normalize(pred)
        tsr = _token_sort_ratio(gt_fuzzy, pred_fuzzy)

        contains_gt = bool(gt_norm) and (gt_norm in pred_norm) and (gt_norm != pred_norm)

        if wer == 0.0 and cer == 0.0:
            label = "verbatim"
        elif contains_gt:
            label = "verbatim_with_extras"
        elif tsr < 30.0:
            label = "total_hallucination"
        else:
            label = "inaccurate_recall"

        return ScoreResult(
            wer=float(wer),
            cer=float(cer),
            chatter_ratio=float(chatter_ratio),
            token_sort_ratio=float(tsr),
            substitutions=int(substitutions),
            deletions=int(deletions),
            insertions=int(insertions),
            ref_words=int(len(gt_words)),
            gt_norm=gt_norm,
            pred_norm=pred_norm,
            label=label,
            contains_gt=bool(contains_gt),
        )
