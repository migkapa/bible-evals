from __future__ import annotations

from statistics import pstdev
from typing import List, Sequence

# Default semantics-preserving paraphrases of "give me this verse's text".
# Used to measure prompt sensitivity: a credible benchmark must show its
# rankings are stable across these, not just under one phrasing.
DEFAULT_PERTURBATIONS = (
    "Quote {ref} from the {version}.",
    "What does {ref} say in the {version}?",
    "Please write out the text of {ref} ({version}).",
    "Recite {ref} from the {version}, word for word.",
)


def verdict_consistency(labels_per_verse: Sequence[Sequence[str]]) -> float:
    """Fraction of verses whose label is identical across all prompt variants.

    1.0 means the model's behavior is fully stable to rephrasing; lower values
    mean the verdict flips depending on how the verse is requested.
    """
    rows = [row for row in labels_per_verse if row]
    if not rows:
        return 1.0
    consistent = sum(1 for row in rows if len(set(row)) == 1)
    return consistent / len(rows)


def per_variant_accuracy(strict_per_verse: Sequence[Sequence[bool]]) -> List[float]:
    """Strict accuracy for each variant, computed column-wise across verses."""
    rows = [row for row in strict_per_verse if row]
    if not rows:
        return []
    n_variants = len(rows[0])
    out = []
    for v in range(n_variants):
        col = [row[v] for row in rows if len(row) > v]
        out.append(sum(1 for x in col if x) / len(col) if col else 0.0)
    return out


def summarize_robustness(
    labels_per_verse: Sequence[Sequence[str]],
    strict_per_verse: Sequence[Sequence[bool]],
) -> dict:
    """Aggregate per-(verse, variant) results into prompt-robustness stats.

    Inputs are matrices indexed [verse][variant]. Returns the per-variant strict
    accuracies, their spread (range + std), the mean, and the verdict-consistency
    rate. ``accuracy_range`` is the headline fragility signal.
    """
    accs = per_variant_accuracy(strict_per_verse)
    n_variants = len(accs)
    acc_range = (max(accs) - min(accs)) if accs else 0.0
    acc_mean = (sum(accs) / len(accs)) if accs else 0.0
    acc_std = pstdev(accs) if len(accs) > 1 else 0.0
    return {
        "n_variants": n_variants,
        "per_variant_strict_accuracy": accs,
        "mean_strict_accuracy": acc_mean,
        "accuracy_range": acc_range,
        "accuracy_std": acc_std,
        "verdict_consistency": verdict_consistency(labels_per_verse),
    }
