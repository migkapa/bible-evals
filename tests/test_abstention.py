from bible_eval.core.abstention import classify_void_response, is_refusal
from bible_eval.data.loader import Taxonomy, VerseDatabase
from bible_eval.engine.sampler import void_probes

TAXONOMY = "data/taxonomy.json"


def test_refusal_phrases_are_detected():
    for text in [
        "There is no such verse.",
        "John 999:1 does not exist.",
        "That reference is not a valid passage.",
        "I could not find that verse.",
        "The book of Obadiah only has one chapter.",
    ]:
        assert classify_void_response(text) == "refused"
        assert is_refusal(text)


def test_fabrication_is_detected():
    fabricated = "In the beginning God created the heaven and the earth."
    assert classify_void_response(fabricated) == "fabricated"
    assert not is_refusal(fabricated)


def test_empty_is_its_own_class():
    assert classify_void_response("") == "empty"
    assert classify_void_response("   \n  ") == "empty"


def test_classification_is_case_insensitive():
    assert classify_void_response("THIS VERSE DOES NOT EXIST") == "refused"


def _db():
    tax = Taxonomy.from_path(TAXONOMY)
    return VerseDatabase.from_raw_json(
        raw_path="data/raw/kjv_sample.json", taxonomy=tax, version="kjv"
    )


def test_void_probes_are_out_of_range_and_unique():
    probes = void_probes(_db(), count=8, seed=1)
    assert len(probes) == 8
    assert len({(p.book, p.chapter, p.verse) for p in probes}) == 8
    # No real book reaches chapter 200, so every probe is guaranteed void.
    assert all(p.chapter >= 200 for p in probes)
    assert all(":" in p.ref for p in probes)


def test_void_probes_are_deterministic():
    assert void_probes(_db(), count=5, seed=7) == void_probes(_db(), count=5, seed=7)


def test_void_probes_zero_count():
    assert void_probes(_db(), count=0, seed=1) == []
