from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenAICompatibleConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "local"
    model: str = "qwen/qwen3.5-9b-instruct"
    timeout_s: float = 6.0
    temperature: float = 0.1
    max_tokens: int = 96


class LocalOpenAICompatibleClient:
    """Small swappable client for local OpenAI-compatible servers."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
        )

    def translate(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        response = self._session.post(url, json=payload, timeout=self.config.timeout_s)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response contained no choices")
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise RuntimeError("LLM response was empty")
        return content

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = self._session.get(f"{self.config.base_url.rstrip('/')}/models", timeout=self.config.timeout_s)
            response.raise_for_status()
        except Exception as exc:
            logger.debug("Local model server healthcheck failed", exc_info=True)
            return False, str(exc)
        return True, "ok"
