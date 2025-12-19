from bible_eval.core.normalizer import NormalizationConfig, Normalizer


def test_normalizer_collapses_whitespace_and_brackets() -> None:
    n = Normalizer(NormalizationConfig())
    assert n.normalize("  Hello   [world]  ") == "Hello world"

