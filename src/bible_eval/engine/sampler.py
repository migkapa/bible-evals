from __future__ import annotations

import random
from dataclasses import dataclass

from bible_eval.data.loader import VerseDatabase, VerseRecord


@dataclass(frozen=True)
class SampleConfig:
    count: int
    seed: int = 1
    stratified: bool = False


@dataclass(frozen=True)
class VoidProbe:
    """A reference to a verse that does not exist (used for abstention probes).

    Quacks like a ``VerseRecord`` for the interrogator's purposes — it only ever
    reads ``.ref`` when building a prompt — but carries no ground-truth text.
    """

    book: str
    chapter: int
    verse: int

    @property
    def ref(self) -> str:
        return f"{self.book} {self.chapter}:{self.verse}"


def tiered_sample(
    db: VerseDatabase, famous_ids: set[int], count: int, seed: int = 1
) -> list[VerseRecord]:
    """Balanced famous/obscure sample for the memorization gradient.

    Draws ~count/2 famous verses (those whose id is in ``famous_ids`` and present
    in the db) and fills the remainder with random obscure verses, so the two
    tiers are comparably sized regardless of how rare the famous set is.
    """
    verses = db.all()
    if not verses or count <= 0:
        return []
    famous = [v for v in verses if v.id in famous_ids]
    obscure = [v for v in verses if v.id not in famous_ids]
    rng = random.Random(seed)
    half = count // 2
    pick_f = min(half, len(famous))
    pick_o = min(count - pick_f, len(obscure))
    out = rng.sample(famous, k=pick_f) + rng.sample(obscure, k=pick_o)
    rng.shuffle(out)
    return out


def void_probes(db: VerseDatabase, count: int, seed: int = 1) -> list[VoidProbe]:
    """Generate references to non-existent verses from books present in ``db``.

    Chapters are drawn from [200, 999] — well beyond any real book (Psalms, the
    longest, has 150) — so every probe is guaranteed not to exist regardless of
    which translation is loaded.
    """
    books = sorted({v.book for v in db.all()})
    if not books or count <= 0:
        return []
    rng = random.Random(seed * 7919 + 13)
    out: list[VoidProbe] = []
    seen: set[tuple[str, int, int]] = set()
    attempts = 0
    while len(out) < count and attempts < count * 50:
        attempts += 1
        book = rng.choice(books)
        chapter = rng.randint(200, 999)
        verse = rng.randint(1, 50)
        key = (book, chapter, verse)
        if key in seen:
            continue
        seen.add(key)
        out.append(VoidProbe(book=book, chapter=chapter, verse=verse))
    return out


class Sampler:
    def __init__(self, cfg: SampleConfig) -> None:
        self.cfg = cfg

    def sample(self, db: VerseDatabase) -> list[VerseRecord]:
        verses = db.all()
        if not verses:
            return []

        rng = random.Random(self.cfg.seed)
        if not self.cfg.stratified or len(verses) <= self.cfg.count:
            return rng.sample(verses, k=min(self.cfg.count, len(verses)))

        ot = [v for v in verses if (v.id // 1_000_000) <= 39]
        nt = [v for v in verses if (v.id // 1_000_000) >= 40]
        half = self.cfg.count // 2
        pick_ot = min(half, len(ot))
        pick_nt = min(self.cfg.count - pick_ot, len(nt))
        out = rng.sample(ot, k=pick_ot) + rng.sample(nt, k=pick_nt)
        rng.shuffle(out)
        return out
