"""Model client abstractions."""

from __future__ import annotations

import time
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


def _coerce_text(value: Any) -> str:
    """Normalize provider-specific message content shapes into text."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)

    return str(value)


@dataclass
class OpenAICompatChatClient:
    """Client for OpenAI-compatible chat completion APIs (vLLM/TGI/Groq gateways)."""

    base_url: str
    model: str
    api_key: str | None = None
    timeout_sec: int = 180
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 2

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

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_sec,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Model request failed after retries: {exc}") from exc

            if response.status_code in {408, 409, 429, 500, 502, 503, 504, 520, 522, 523, 524}:
                last_error = RuntimeError(
                    f"Transient model backend status {response.status_code}: {response.text[:200]}"
                )
                if attempt < self.max_retries:
                    time.sleep(0.7 * (attempt + 1))
                    continue

            if response.status_code >= 400:
                # Handle provider-specific tool failure by retrying with stricter extraction settings.
                try:
                    err = response.json().get("error", {})
                except Exception:
                    err = {}

                code = (err.get("code") or "").lower()
                message = str(err.get("message") or response.text[:300])
                can_retry_tool_failure = (
                    attempt < self.max_retries and code == "tool_use_failed"
                )
                if can_retry_tool_failure:
                    payload["max_tokens"] = max(int(payload.get("max_tokens", max_tokens)), max_tokens * 2)
                    payload["reasoning_effort"] = "low"
                    payload["messages"][0]["content"] = (
                        system_prompt
                        + "\n\nRetry note: output plain text only and end with FINAL_ANSWER line."
                    )
                    time.sleep(0.6 * (attempt + 1))
                    continue

                response.raise_for_status()
                raise RuntimeError(f"Model request failed ({response.status_code}): {message}")

            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                last_error = RuntimeError("Model response missing choices")
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break

            message = choices[0].get("message") or {}
            content = _coerce_text(message.get("content")).strip()
            if content:
                return content

            # Some hosted backends put long text into `reasoning` when `content` is empty.
            reasoning = _coerce_text(message.get("reasoning")).strip()
            if reasoning:
                return reasoning

            executed_tools = message.get("executed_tools")
            if executed_tools:
                tool_text = _coerce_text(executed_tools).strip()
                if tool_text:
                    return tool_text

            last_error = RuntimeError("Model response had empty content and no fallback fields")
            if attempt < self.max_retries:
                payload["max_tokens"] = max(int(payload.get("max_tokens", max_tokens)), max_tokens * 2)
                time.sleep(0.5 * (attempt + 1))
                continue

        raise RuntimeError(str(last_error or "Model generation failed"))
