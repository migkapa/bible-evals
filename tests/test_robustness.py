from bible_eval.core.robustness import (
    per_variant_accuracy,
    summarize_robustness,
    verdict_consistency,
)


def test_verdict_consistency_all_stable():
    labels = [["verbatim", "verbatim"], ["inaccurate_recall", "inaccurate_recall"]]
    assert verdict_consistency(labels) == 1.0


def test_verdict_consistency_partial():
    labels = [
        ["verbatim", "verbatim"],  # stable
        ["verbatim", "inaccurate_recall"],  # flips
    ]
    assert verdict_consistency(labels) == 0.5


def test_verdict_consistency_empty_is_vacuously_stable():
    assert verdict_consistency([]) == 1.0


def test_per_variant_accuracy_is_columnwise():
    # 2 verses x 3 variants
    strict = [
        [True, False, True],
        [True, True, False],
    ]
    # variant0: 2/2, variant1: 1/2, variant2: 1/2
    assert per_variant_accuracy(strict) == [1.0, 0.5, 0.5]


def test_summarize_robustness_reports_spread():
    labels = [["verbatim", "inaccurate_recall"], ["verbatim", "verbatim"]]
    strict = [[True, False], [True, True]]
    s = summarize_robustness(labels, strict)
    assert s["n_variants"] == 2
    assert s["per_variant_strict_accuracy"] == [1.0, 0.5]
    assert s["accuracy_range"] == 0.5
    assert s["mean_strict_accuracy"] == 0.75
    assert s["verdict_consistency"] == 0.5
    assert s["accuracy_std"] > 0.0


def test_summarize_robustness_empty():
    s = summarize_robustness([], [])
    assert s["n_variants"] == 0
    assert s["accuracy_range"] == 0.0
    assert s["verdict_consistency"] == 1.0
