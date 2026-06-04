"""LLM provider construction from settings (key sourced from the environment)."""

from __future__ import annotations

import pytest

from market_trader.config import Settings
from market_trader.reasoning import (
    AnthropicProvider,
    LLMError,
    anthropic_provider_from_settings,
)


def test_provider_from_settings_requires_a_key() -> None:
    with pytest.raises(LLMError):
        anthropic_provider_from_settings(Settings(anthropic_api_key=None))


def test_provider_from_settings_builds_client_when_key_present() -> None:
    provider = anthropic_provider_from_settings(
        Settings(anthropic_api_key="sk-test-not-real", anthropic_model="claude-opus-4-8")
    )
    assert isinstance(provider, AnthropicProvider)  # constructed; no network until .complete()
