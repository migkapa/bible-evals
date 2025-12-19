from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bible_eval.connectors.base import GenerateRequest
from bible_eval.data.loader import VerseRecord


@dataclass(frozen=True)
class ReferenceOptions:
    """
    Baseline connector for offline / smoke testing.

    modes:
      - verbatim: return the ground-truth verse text exactly
      - chatter: add a preface/suffix around the verse
      - empty: return an empty string
    """

    mode: str = "verbatim"
    prefix: str = "Here is the verse: "
    suffix: str = ""


class ReferenceConnector:
    def __init__(self, opts: Optional[ReferenceOptions] = None) -> None:
        self.opts = opts or ReferenceOptions()

    def generate(self, req: GenerateRequest) -> str:
        raise RuntimeError("ReferenceConnector requires generate_for_verse()")

    def generate_for_verse(self, verse: VerseRecord, req: GenerateRequest) -> str:
        mode = self.opts.mode
        if mode == "verbatim":
            return verse.text
        if mode == "chatter":
            return f"{self.opts.prefix}{verse.text}{self.opts.suffix}".strip()
        if mode == "empty":
            return ""
        raise ValueError(f"Unknown reference mode: {mode!r}")

