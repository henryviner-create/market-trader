"""Structured signal extraction from text via an LLM — the breadth factory's core.

Opus reads public text (e.g. recent headlines for a name) and returns a *structured,
bounded* score, which becomes a candidate Feature that must still clear the IC gate like
any other signal. Two hard rules keep it safe:

* **point-in-time** — the model is told to use ONLY the provided text and never its own
  knowledge of what happened next (no lookahead via training data); the resulting
  observation's ``knowledge_time`` is the document date, set by the collector.
* **fail-soft** — malformed / unparseable output yields ``None``, never a guessed number,
  so a bad extraction is a *missing* (neutral) signal, not a wrong one.

The LLM never sizes or triggers a trade; it only proposes a number for the validated,
risk-gated pipeline to (dis)trust on measured IC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from market_trader.observability import get_logger
from market_trader.reasoning.budget import LLMBudgetExceeded
from market_trader.reasoning.llm import LLMProvider

_log = get_logger("llm_extract")

NEWS_SIGNAL_SYSTEM = (
    "You are a careful equity analyst extracting ONE bounded sentiment signal from news "
    "headlines about a single company. Use ONLY the provided headlines. Do not use outside "
    "knowledge, and never reason about what happened after these headlines. Judge "
    "materiality: ignore routine or promotional items. Respond with STRICT JSON only, no "
    'prose: {"score": <float -1..1>, "confidence": <float 0..1>, "rationale": <short '
    "string>}. score: -1 strongly bearish, 0 neutral/immaterial, +1 strongly bullish. If "
    "the headlines are immaterial or absent, return score 0 with low confidence."
)


@dataclass(frozen=True)
class ExtractedSignal:
    symbol: str
    score: float  # bounded to [-1, 1]
    confidence: float  # bounded to [0, 1]
    rationale: str


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_signal(symbol: str, text: str) -> ExtractedSignal | None:
    """Parse the model's JSON, tolerating code fences / prose around it; reject malformed.

    Takes the first ``{...}`` block so a fenced or prefaced reply still parses; any missing
    field, bad type, or non-JSON yields ``None`` (a neutral, missing signal).
    """
    s = text.strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
        score = _clip(float(obj["score"]), -1.0, 1.0)
        confidence = _clip(float(obj.get("confidence", 0.5)), 0.0, 1.0)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return ExtractedSignal(symbol, score, confidence, str(obj.get("rationale", ""))[:300])


def extract_news_signal(
    provider: LLMProvider, *, symbol: str, headlines: list[str], max_headlines: int = 20
) -> ExtractedSignal | None:
    """Extract a bounded sentiment signal for ``symbol`` from its recent ``headlines``.

    Returns ``None`` when there is nothing to read or the output can't be parsed — a
    missing (neutral) signal, never a guessed one. A budget-exhaustion propagates so the
    caller's batch can stop; any other provider error is logged and swallowed (skip this
    name, keep the batch alive).
    """
    clean = [h.strip() for h in headlines if h and h.strip()][:max_headlines]
    if not clean:
        return None
    prompt = f"Company ticker: {symbol}\nRecent headlines:\n" + "\n".join(f"- {h}" for h in clean)
    try:
        text = provider.complete(system=NEWS_SIGNAL_SYSTEM, prompt=prompt, max_tokens=300)
    except LLMBudgetExceeded:
        raise  # let the batch stop cleanly rather than silently neutralising every name
    except Exception as exc:  # one bad call must not abort the sweep
        _log.warning("llm_extract_failed", symbol=symbol, error=str(exc))
        return None
    signal = parse_signal(symbol, text)
    if signal is None:
        _log.warning("llm_extract_unparseable", symbol=symbol)
    return signal
