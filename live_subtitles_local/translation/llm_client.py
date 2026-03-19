from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model_name: str, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_s = timeout_s
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout_s) if OpenAI else None

    def translate_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str | None:
        try:
            if self._client is not None:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""
                return content.strip()
            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}
            if httpx is None:
                raise RuntimeError("httpx is not installed.")
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception as exc:  # pragma: no cover - network errors are environment specific.
            logger.warning("Translation request failed: %s", exc)
            return None
