from bible_eval.core.popularity import (
    classify,
    load_famous_ids,
    parse_ref,
    split_by_tier,
    summarize_gradient,
)
from bible_eval.data.loader import Taxonomy
from bible_eval.utils.verse_id import VerseIdCodec

TAX = Taxonomy.from_path("data/taxonomy.json")
CODEC = VerseIdCodec()


def test_parse_ref_handles_numbered_books():
    # 1 Corinthians 13:4 -> book 46, chapter 13, verse 4
    vid = parse_ref("1 Corinthians 13:4", TAX, CODEC)
    assert CODEC.decode(vid).chapter == 13
    assert CODEC.decode(vid).verse == 4
    # John 3:16 resolves too
    assert parse_ref("John 3:16", TAX, CODEC) == CODEC.encode(
        book_index=TAX.book_index("John"), chapter=3, verse=16
    )


def test_load_famous_ids_from_repo_file():
    ids = load_famous_ids("data/popularity.json", TAX, CODEC)
    assert len(ids) >= 50
    assert parse_ref("John 3:16", TAX, CODEC) in ids


def test_classify_and_split():
    famous = {parse_ref("John 3:16", TAX, CODEC)}
    j316 = parse_ref("John 3:16", TAX, CODEC)
    other = parse_ref("Genesis 5:5", TAX, CODEC)
    assert classify(j316, famous) == "famous"
    assert classify(other, famous) == "obscure"
    f, o = split_by_tier([j316, other], famous)
    assert f == [j316] and o == [other]


def test_summarize_gradient_computes_gap():
    per_verse = [
        ("famous", True), ("famous", True), ("famous", False),  # 2/3
        ("obscure", False), ("obscure", False), ("obscure", True),  # 1/3
    ]
    g = summarize_gradient(per_verse)
    assert g["famous"]["strict_accuracy"] == 2 / 3
    assert g["obscure"]["strict_accuracy"] == 1 / 3
    assert abs(g["gap"] - (2 / 3 - 1 / 3)) < 1e-9


def test_summarize_gradient_handles_missing_tier():
    g = summarize_gradient([("famous", True)])
    assert g["obscure"]["strict_accuracy"] is None
    assert g["gap"] is None
