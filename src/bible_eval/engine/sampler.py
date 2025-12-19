from __future__ import annotations

import random
from dataclasses import dataclass

from bible_eval.data.loader import VerseDatabase, VerseRecord


@dataclass(frozen=True)
class SampleConfig:
    count: int
    seed: int = 1
    stratified: bool = False


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
