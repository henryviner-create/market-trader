"""LLM provider abstraction.

Production calls the hosted Anthropic API (no local LLM). To keep the engine image
lean we talk to the Messages API over stdlib ``urllib`` rather than pulling an SDK;
the abstraction means that's swappable. Tests use :class:`MockLLMProvider` — no
network, fully deterministic.
"""

from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod

from market_trader.config import Settings
from market_trader.observability import get_logger

_log = get_logger("llm")

ANTHROPIC_VERSION = "2023-06-01"


class LLMError(RuntimeError):
    pass


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


class MockLLMProvider(LLMProvider):
    """Deterministic provider for tests/offline use. Records the calls it receives."""

    def __init__(self, canned: str = "MOCK BRIEF") -> None:
        self.canned = canned
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.calls.append((system, prompt))
        return self.canned


class AnthropicProvider(LLMProvider):
    """Minimal hosted Anthropic Messages client (no SDK dependency)."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-opus-4-8",
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise LLMError("Anthropic API key is required (MT_ANTHROPIC_API_KEY).")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        body = json.dumps(
            {
                "model": self._model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        request = urllib.request.Request(
            f"{self._base_url}/v1/messages",
            data=body,
            method="POST",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                payload = json.loads(resp.read())
        except Exception as exc:  # network/HTTP errors are surfaced, never swallowed
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        usage = payload.get("usage", {})
        _log.info(
            "llm_call",
            model=self._model,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        blocks = payload.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if not text:
            raise LLMError("Anthropic response contained no text content.")
        return text


def anthropic_provider_from_settings(settings: Settings) -> AnthropicProvider:
    """Build a live Anthropic client from configured settings.

    The key comes from the environment (``MT_ANTHROPIC_API_KEY``) via ``Settings`` —
    never from the repo or image. Raises :class:`LLMError` if it is not set.
    """
    if not settings.anthropic_api_key:
        raise LLMError("MT_ANTHROPIC_API_KEY is not set; cannot build an Anthropic client")
    return AnthropicProvider(settings.anthropic_api_key, model=settings.anthropic_model)
