from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

from bible_eval.connectors.base import Connector, GenerateRequest


@dataclass(frozen=True)
class OpenAICompatibleOptions:
    base_url: str = "http://localhost:8000"
    model: str = "local-model"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    timeout_s: float = 120.0


class OpenAICompatibleConnector(Connector):
    def __init__(self, opts: OpenAICompatibleOptions) -> None:
        self.opts = opts

    def generate(self, req: GenerateRequest) -> str:
        url = f"{self.opts.base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.opts.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.user})

        body = {"model": self.opts.model, "messages": messages, "temperature": self.opts.temperature}
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=self.opts.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return str(msg.get("content", "")).strip()
