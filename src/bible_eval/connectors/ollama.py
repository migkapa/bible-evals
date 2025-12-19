from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any
from typing import Optional

from bible_eval.connectors.base import Connector, GenerateRequest


@dataclass(frozen=True)
class OllamaOptions:
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    temperature: float = 0.0
    num_predict: Optional[int] = None
    think: Optional[bool] = None
    strip_thinking: bool = False
    timeout_s: float = 120.0


class OllamaConnector(Connector):
    def __init__(self, opts: OllamaOptions) -> None:
        self.opts = opts

    def generate(self, req: GenerateRequest) -> str:
        url = f"{self.opts.base_url.rstrip('/')}/api/chat"
        messages: list[dict[str, str]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.user})

        options: dict[str, Any] = {"temperature": self.opts.temperature}
        if self.opts.num_predict is not None:
            options["num_predict"] = self.opts.num_predict

        body = {
            "model": self.opts.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if self.opts.think is not None:
            body["think"] = bool(self.opts.think)
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.opts.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return str(payload.get("message", {}).get("content", "")).strip()
