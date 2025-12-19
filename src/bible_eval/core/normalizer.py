from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_SMART_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
}


@dataclass(frozen=True)
class NormalizationConfig:
    strip_brackets: bool = True
    strip_pilcrow: bool = True
    standardize_quotes: bool = True
    collapse_whitespace: bool = True
    casefold: bool = False


class Normalizer:
    def __init__(self, cfg: Optional[NormalizationConfig] = None) -> None:
        self.cfg = cfg or NormalizationConfig()

    def normalize(self, text: str) -> str:
        out = text
        if self.cfg.standardize_quotes:
            out = "".join(_SMART_QUOTES.get(ch, ch) for ch in out)
        if self.cfg.strip_pilcrow:
            out = out.replace("¶", "")
        if self.cfg.strip_brackets:
            out = re.sub(r"[\[\]]", "", out)
        if self.cfg.collapse_whitespace:
            out = re.sub(r"\s+", " ", out).strip()
        if self.cfg.casefold:
            out = out.casefold()
        return out
