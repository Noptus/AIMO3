"""Model client abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import requests


class ChatClient(Protocol):
    """Minimal protocol for chat-completions backends."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        ...


@dataclass
class OpenAICompatChatClient:
    """Client for OpenAI-compatible chat completion APIs (vLLM/TGI gateways)."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout_sec: int = 180
    extra_body: dict[str, Any] = field(default_factory=dict)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(self.extra_body)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Model response missing choices")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise RuntimeError("Model response missing message content")

        return content
