"""Reasoning tier (LLM): synthesis and the daily briefing.

Runs *alongside* the quant ensemble, never inside it. The briefing is constrained
by the data — it may only assert what the signals/flow support, and must state the
case against. It is an input to judgement, never a verdict.
"""

from market_trader.reasoning.briefing import (
    BriefingContext,
    build_briefing_context,
    generate_llm_brief,
    render_brief_markdown,
)
from market_trader.reasoning.llm import AnthropicProvider, LLMError, LLMProvider, MockLLMProvider

__all__ = [
    "AnthropicProvider",
    "BriefingContext",
    "LLMError",
    "LLMProvider",
    "MockLLMProvider",
    "build_briefing_context",
    "generate_llm_brief",
    "render_brief_markdown",
]
