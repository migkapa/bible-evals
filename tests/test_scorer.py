from bible_eval.core.scorer import ErrorCategory, Scorer


def test_scorer_perfect_match() -> None:
    s = Scorer()
    r = s.score_pair("Jesus wept.", "Jesus wept.")
    assert r.wer == 0.0
    assert r.cer == 0.0
    assert r.chatter_ratio == 0.0
    assert r.label == "verbatim"
    assert r.error_category == ErrorCategory.VERBATIM
    assert r.semantic_similarity == 1.0


def test_scorer_insertion_penalized() -> None:
    s = Scorer()
    r = s.score_pair("Jesus wept.", "And Jesus wept.")
    assert r.wer > 0.0
    assert r.chatter_ratio > 0.0
    assert r.insertion_ratio > 0.0


def test_scorer_verbatim_with_extras_when_contains_gt() -> None:
    s = Scorer()
    r = s.score_pair("Jesus wept.", '"Jesus wept." - John 11:35 (KJV)')
    assert r.contains_gt is True
    assert r.label == "verbatim_with_extras"
    assert r.error_category == ErrorCategory.VERBATIM_WITH_EXTRAS


def test_scorer_minor_deviation() -> None:
    """Small typo should be classified as minor deviation."""
    s = Scorer()
    # One character difference in a longer text
    gt = "In the beginning God created the heaven and the earth."
    pred = "In the beginning God created the heavens and the earth."  # heaven -> heavens
    r = s.score_pair(gt, pred)
    assert r.cer < 0.05
    assert r.error_category == ErrorCategory.MINOR_DEVIATION


def test_scorer_omission() -> None:
    """Significant deletions should be classified as omission."""
    s = Scorer()
    gt = "In the beginning God created the heaven and the earth."
    pred = "In the beginning God created."  # Truncated
    r = s.score_pair(gt, pred)
    assert r.deletion_ratio > 0.20
    assert r.error_category == ErrorCategory.OMISSION


def test_scorer_total_hallucination() -> None:
    """Completely different content should be total hallucination."""
    s = Scorer()
    gt = "Jesus wept."
    pred = "The quick brown fox jumps over the lazy dog."
    r = s.score_pair(gt, pred)
    assert r.token_sort_ratio < 30.0
    assert r.label == "total_hallucination"
    assert r.error_category == ErrorCategory.TOTAL_HALLUCINATION


def test_scorer_semantic_similarity() -> None:
    """Semantic similarity should be higher for related content."""
    s = Scorer()

    # Same content = perfect similarity
    r1 = s.score_pair("God created heaven earth", "God created heaven earth")
    assert r1.semantic_similarity == 1.0

    # Some overlap
    r2 = s.score_pair("God created heaven earth", "God made heaven world")
    assert 0.3 < r2.semantic_similarity < 0.9

    # No overlap
    r3 = s.score_pair("Jesus wept", "The fox jumped")
    assert r3.semantic_similarity < 0.3


def test_scorer_deletion_substitution_insertion_ratios() -> None:
    """Test that edit ratios are computed correctly."""
    s = Scorer()
    gt = "one two three four five"  # 5 words
    pred = "one TWO four five six"  # substituted 'two'->TWO, deleted 'three', inserted 'six'
    r = s.score_pair(gt, pred)
    # Should have some of each type of error
    assert r.ref_words == 5
    assert r.substitutions + r.deletions + r.insertions > 0
