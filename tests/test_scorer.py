from bible_eval.core.scorer import Scorer


def test_scorer_perfect_match() -> None:
    s = Scorer()
    r = s.score_pair("Jesus wept.", "Jesus wept.")
    assert r.wer == 0.0
    assert r.cer == 0.0
    assert r.chatter_ratio == 0.0
    assert r.label == "verbatim"


def test_scorer_insertion_penalized() -> None:
    s = Scorer()
    r = s.score_pair("Jesus wept.", "And Jesus wept.")
    assert r.wer > 0.0
    assert r.chatter_ratio > 0.0


def test_scorer_verbatim_with_extras_when_contains_gt() -> None:
    s = Scorer()
    r = s.score_pair("Jesus wept.", '"Jesus wept." - John 11:35 (KJV)')
    assert r.contains_gt is True
    assert r.label == "verbatim_with_extras"
