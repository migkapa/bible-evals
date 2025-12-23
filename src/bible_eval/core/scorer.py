from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import List, Optional

from bible_eval.core.normalizer import NormalizationConfig, Normalizer


# ---------------------------------------------------------------------------
# Error Category Definitions
# ---------------------------------------------------------------------------
# More granular error classification for better analysis

class ErrorCategory:
    """Detailed error categorization for non-verbatim responses."""

    VERBATIM = "verbatim"  # Perfect match
    VERBATIM_WITH_EXTRAS = "verbatim_with_extras"  # Correct text + quotes/citations

    # Subcategories of inaccurate recall (from least to most severe)
    MINOR_DEVIATION = "minor_deviation"  # Small typos/punctuation (CER < 5%, WER < 10%)
    OMISSION = "omission"  # Significant deletions (deletions > 20% of words)
    PARAPHRASE = "paraphrase"  # Meaning preserved but words changed (TSR >= 70%)
    PARTIAL_RECALL = "partial_recall"  # Some correct content mixed with errors

    # Severe errors
    TOTAL_HALLUCINATION = "total_hallucination"  # Completely wrong (TSR < 30%)

    @classmethod
    def all_categories(cls) -> List[str]:
        return [
            cls.VERBATIM,
            cls.VERBATIM_WITH_EXTRAS,
            cls.MINOR_DEVIATION,
            cls.OMISSION,
            cls.PARAPHRASE,
            cls.PARTIAL_RECALL,
            cls.TOTAL_HALLUCINATION,
        ]


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
    # New: detailed error category and semantic similarity
    error_category: str = ""
    semantic_similarity: float = 0.0
    deletion_ratio: float = 0.0  # deletions / ref_words
    substitution_ratio: float = 0.0  # substitutions / ref_words
    insertion_ratio: float = 0.0  # insertions / ref_words


def _compute_semantic_similarity(gt_words: List[str], pred_words: List[str]) -> float:
    """
    Compute semantic similarity using word overlap with IDF-like weighting.

    This is a lightweight alternative to embedding-based similarity that:
    - Gives higher weight to rare/distinctive words
    - Penalizes missing key content words
    - Works without external dependencies

    Returns a score between 0.0 and 1.0.
    """
    if not gt_words or not pred_words:
        return 0.0 if gt_words or pred_words else 1.0

    # Common stop words to downweight
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
        "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
        "our", "their", "what", "which", "who", "whom", "whose", "where", "when",
        "why", "how", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "not", "only", "same", "so", "than",
        "too", "very", "just", "also", "now", "here", "there", "then", "once",
    }

    def word_weight(word: str) -> float:
        """Assign weight to word - content words get higher weight."""
        w = word.lower()
        if w in stop_words:
            return 0.3
        if len(w) <= 2:
            return 0.5
        return 1.0

    gt_set = set(w.lower() for w in gt_words)
    pred_set = set(w.lower() for w in pred_words)

    # Weighted intersection
    intersection = gt_set & pred_set
    weighted_overlap = sum(word_weight(w) for w in intersection)

    # Weighted ground truth (what we expect)
    weighted_gt = sum(word_weight(w) for w in gt_set)

    if weighted_gt == 0:
        return 1.0 if not pred_set else 0.0

    # Recall-oriented: how much of the expected content was found
    recall = weighted_overlap / weighted_gt

    # Penalize excessive additions (precision component)
    weighted_pred = sum(word_weight(w) for w in pred_set)
    precision = weighted_overlap / weighted_pred if weighted_pred > 0 else 0.0

    # F1-like combination favoring recall (content preservation is more important)
    if recall + precision == 0:
        return 0.0
    f_beta = (1.25 * precision * recall) / (0.25 * precision + recall)  # beta=0.5

    return min(1.0, max(0.0, f_beta))


def _classify_error(
    wer: float,
    cer: float,
    tsr: float,
    deletion_ratio: float,
    substitution_ratio: float,
    insertion_ratio: float,
    contains_gt: bool,
    semantic_sim: float,
) -> str:
    """
    Classify the error into a detailed category.

    Categories (from best to worst):
    - verbatim: Perfect match
    - verbatim_with_extras: Correct text with added quotes/citations
    - minor_deviation: Small typos/punctuation differences
    - omission: Significant content deleted
    - paraphrase: Meaning preserved but wording changed
    - partial_recall: Mixed correct and incorrect content
    - total_hallucination: Completely wrong content
    """
    # Perfect match
    if wer == 0.0 and cer == 0.0:
        return ErrorCategory.VERBATIM

    # Contains full ground truth with extras
    if contains_gt:
        return ErrorCategory.VERBATIM_WITH_EXTRAS

    # Total hallucination - very low token overlap
    if tsr < 30.0:
        return ErrorCategory.TOTAL_HALLUCINATION

    # Minor deviation - small character-level errors, high TSR
    if cer < 0.05 and wer <= 0.15 and tsr >= 90.0:
        return ErrorCategory.MINOR_DEVIATION

    # Omission - primarily deletions (truncated output)
    if deletion_ratio > 0.25 and substitution_ratio <= 0.15:
        return ErrorCategory.OMISSION

    # Paraphrase - high semantic similarity but different words
    if tsr >= 70.0 and semantic_sim >= 0.7:
        return ErrorCategory.PARAPHRASE

    # Partial recall - everything else
    return ErrorCategory.PARTIAL_RECALL


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
        ref_word_count = len(gt_words)
        wer = (distance / ref_word_count) if ref_word_count else (0.0 if not pred_words else 1.0)

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

        # Compute ratios for detailed categorization
        deletion_ratio = deletions / ref_word_count if ref_word_count else 0.0
        substitution_ratio = substitutions / ref_word_count if ref_word_count else 0.0
        insertion_ratio = insertions / ref_word_count if ref_word_count else 0.0

        # Compute semantic similarity
        semantic_sim = _compute_semantic_similarity(gt_words, pred_words)

        # Detailed error categorization
        error_category = _classify_error(
            wer=wer,
            cer=cer,
            tsr=tsr,
            deletion_ratio=deletion_ratio,
            substitution_ratio=substitution_ratio,
            insertion_ratio=insertion_ratio,
            contains_gt=contains_gt,
            semantic_sim=semantic_sim,
        )

        # Legacy label for backwards compatibility
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
            ref_words=int(ref_word_count),
            gt_norm=gt_norm,
            pred_norm=pred_norm,
            label=label,
            contains_gt=bool(contains_gt),
            error_category=error_category,
            semantic_similarity=float(semantic_sim),
            deletion_ratio=float(deletion_ratio),
            substitution_ratio=float(substitution_ratio),
            insertion_ratio=float(insertion_ratio),
        )
