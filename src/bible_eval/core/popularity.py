from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

from bible_eval.data.loader import Taxonomy
from bible_eval.utils.verse_id import VerseIdCodec

FAMOUS = "famous"
OBSCURE = "obscure"


def parse_ref(ref: str, taxonomy: Taxonomy, codec: VerseIdCodec) -> int:
    """Resolve a 'Book Chapter:Verse' reference to a canonical verse id.

    Handles multi-word / numbered book names ('1 Corinthians 13:4') by splitting
    off the trailing 'C:V' token and resolving the remainder via the taxonomy.
    """
    book_part, cv = ref.strip().rsplit(" ", 1)
    chapter_s, verse_s = cv.split(":")
    book_index = taxonomy.book_index(book_part)
    return codec.encode(book_index=book_index, chapter=int(chapter_s), verse=int(verse_s))


def load_famous_ids(path: str, taxonomy: Taxonomy, codec: VerseIdCodec | None = None) -> Set[int]:
    """Load the curated 'famous' verse set as canonical ids."""
    codec = codec or VerseIdCodec()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    refs = data.get("famous", []) if isinstance(data, dict) else data
    return {parse_ref(r, taxonomy, codec) for r in refs}


def classify(verse_id: int, famous_ids: Set[int]) -> str:
    return FAMOUS if verse_id in famous_ids else OBSCURE


def summarize_gradient(per_verse: Sequence[Tuple[str, bool]]) -> dict:
    """Per-tier strict accuracy and the famous−obscure gap.

    ``per_verse`` is a sequence of (tier, strict_hit) pairs. The gap is the
    headline signal: a large positive gap means recall tracks popularity, i.e.
    memorization rather than uniform competence.
    """
    def acc(tier: str) -> Tuple[int, int]:
        hits = sum(1 for t, s in per_verse if t == tier and s)
        n = sum(1 for t, _ in per_verse if t == tier)
        return hits, n

    f_hits, f_n = acc(FAMOUS)
    o_hits, o_n = acc(OBSCURE)
    f_acc = (f_hits / f_n) if f_n else None
    o_acc = (o_hits / o_n) if o_n else None
    gap = (f_acc - o_acc) if (f_acc is not None and o_acc is not None) else None
    return {
        "famous": {"n": f_n, "hits": f_hits, "strict_accuracy": f_acc},
        "obscure": {"n": o_n, "hits": o_hits, "strict_accuracy": o_acc},
        "gap": gap,
    }


def split_by_tier(verse_ids: Iterable[int], famous_ids: Set[int]) -> Tuple[List[int], List[int]]:
    """Partition verse ids into (famous, obscure) lists, preserving order."""
    famous: List[int] = []
    obscure: List[int] = []
    for vid in verse_ids:
        (famous if vid in famous_ids else obscure).append(vid)
    return famous, obscure
