from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from bible_eval.connectors.base import GenerateRequest
from bible_eval.connectors.ollama import OllamaConnector, OllamaOptions
from bible_eval.connectors.openai_compatible import OpenAICompatibleConnector, OpenAICompatibleOptions
from bible_eval.connectors.reference import ReferenceConnector, ReferenceOptions
from bible_eval.data.loader import VerseRecord


@dataclass(frozen=True)
class PromptSet:
    naive_user: str
    constraint_user: str
    system2_system: str
    system2_user: str


class Interrogator:
    def __init__(
        self,
        connector,
        model_name: str,
        prompts: PromptSet,
        prompt_mode: str,
        *,
        max_retries: int = 2,
        backoff_s: float = 1.0,
    ) -> None:
        self.connector = connector
        self.model_name = model_name
        self.prompts = prompts
        self.prompt_mode = prompt_mode
        self.max_retries = max_retries
        self.backoff_s = backoff_s

    @classmethod
    def prompts_from_config(cls, cfg: dict) -> PromptSet:
        p = cfg["prompts"]
        return PromptSet(
            naive_user=p["naive_user"],
            constraint_user=p["constraint_user"],
            system2_system=p["system2_system"],
            system2_user=p["system2_user"],
        )

    @classmethod
    def from_model_config(cls, cfg: dict, *, model_cfg: dict, prompt_mode: str) -> "Interrogator":
        connector_name = model_cfg["connector"]
        model_name = model_cfg["name"]
        opts = model_cfg.get("options", {})

        if connector_name == "ollama":
            connector = OllamaConnector(OllamaOptions(**opts))
        elif connector_name in {"openai-compatible", "openai_compatible"}:
            connector = OpenAICompatibleConnector(OpenAICompatibleOptions(**opts))
        elif connector_name in {"reference", "baseline"}:
            connector = ReferenceConnector(ReferenceOptions(**opts))
        else:
            raise ValueError(f"Unknown connector: {connector_name!r}")

        engine_cfg = cfg.get("engine", {}) or {}
        prompts = cls.prompts_from_config(cfg)
        return cls(
            connector=connector,
            model_name=model_name,
            prompts=prompts,
            prompt_mode=prompt_mode,
            max_retries=int(engine_cfg.get("max_retries", 2)),
            backoff_s=float(engine_cfg.get("backoff_s", 1.0)),
        )

    def _build(self, ref: str, version_name: str) -> GenerateRequest:
        if self.prompt_mode == "naive":
            return GenerateRequest(system=None, user=self.prompts.naive_user.format(ref=ref, version=version_name))
        if self.prompt_mode == "constraint":
            return GenerateRequest(
                system=None, user=self.prompts.constraint_user.format(ref=ref, version=version_name)
            )
        if self.prompt_mode == "system2":
            return GenerateRequest(
                system=self.prompts.system2_system,
                user=self.prompts.system2_user.format(ref=ref, version=version_name),
            )
        raise ValueError(f"Unknown prompt_mode: {self.prompt_mode!r}")

    def build_request(self, *, verse: VerseRecord, version_name: str) -> GenerateRequest:
        return self._build(ref=verse.ref, version_name=version_name)

    def _generate(self, verse: VerseRecord, req: GenerateRequest) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                if hasattr(self.connector, "generate_for_verse"):
                    return self.connector.generate_for_verse(verse, req)  # type: ignore[attr-defined]
                return self.connector.generate(req)
            except Exception as e:  # noqa: BLE001 - boundary with external connectors
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_s * (2**attempt))
        raise RuntimeError(f"Model query failed after retries: {self.model_name}") from last_err

    def query(self, verse: VerseRecord, version_name: str) -> str:
        req = self.build_request(verse=verse, version_name=version_name)
        return self._generate(verse, req)

    def query_with_request(self, verse: VerseRecord, version_name: str) -> tuple[GenerateRequest, str]:
        req = self.build_request(verse=verse, version_name=version_name)
        return req, self._generate(verse, req)

    def query_with_latency(self, verse: VerseRecord, version_name: str) -> tuple[GenerateRequest, str, float]:
        """
        Query the model and return the request, response, and latency in seconds.

        Returns:
            Tuple of (request, response, latency_seconds)
        """
        req = self.build_request(verse=verse, version_name=version_name)
        start_time = time.perf_counter()
        response = self._generate(verse, req)
        latency = time.perf_counter() - start_time
        return req, response, latency
