import pytest

from bible_eval.data.fetch import normalize_scrollmapper, source_url
from bible_eval.data.loader import Taxonomy

TAX = Taxonomy.from_path("data/taxonomy.json")

SAMPLE = {
    "translation": "Test",
    "books": [
        {
            "name": "Genesis",
            "chapters": [
                {"chapter": 1, "verses": [{"verse": 1, "text": "In the beginning"}, {"verse": 2, "text": "  "}]}
            ],
        },
        {
            "name": "Revelation of John",  # alias must resolve to Revelation
            "chapters": [{"chapter": 22, "verses": [{"verse": 21, "text": "Amen."}]}],
        },
    ],
}


def test_normalize_flattens_and_skips_empty():
    out = normalize_scrollmapper(SAMPLE, TAX)
    # 3 verses present, but the whitespace-only one is dropped.
    assert len(out) == 2
    assert out[0] == {"book": "Genesis", "chapter": 1, "verse": 1, "text": "In the beginning"}
    assert out[-1]["book"] == "Revelation of John"


def test_normalize_rejects_unknown_book():
    bad = {"books": [{"name": "Nephi", "chapters": [{"chapter": 1, "verses": [{"verse": 1, "text": "x"}]}]}]}
    with pytest.raises(ValueError, match="not in taxonomy"):
        normalize_scrollmapper(bad, TAX)


def test_normalize_rejects_empty_source():
    with pytest.raises(ValueError, match="books"):
        normalize_scrollmapper({"books": []}, TAX)


def test_source_url_known_and_unknown():
    assert source_url("kjv").endswith("/KJV.json")
    assert source_url("ASV").endswith("/ASV.json")
    with pytest.raises(ValueError, match="No known public-domain source"):
        source_url("nrsv")
