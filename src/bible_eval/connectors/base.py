from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GenerateRequest:
    system: Optional[str]
    user: str


class Connector:
    def generate(self, req: GenerateRequest) -> str:
        raise NotImplementedError
