"""Portfolio & risk layer — where risk-adjusted returns are actually generated.

* **weighting** — conditional signal weighting (rolling IC, decay, orthogonality)
  into a composite score that is a *ranking input*, never a direct trade trigger.
* **construction** — vol-targeting, fractional Kelly, risk-parity, HRP, and
  Ledoit-Wolf-shrunk mean-variance.
* **risk** — hard position/exposure limits (over-limit orders are *refused in
  code*) and a drawdown circuit-breaker, sitting above execution.
"""

from market_trader.portfolio.construction import (
    fractional_kelly_weights,
    hierarchical_risk_parity,
    ledoit_wolf_cov,
    min_variance_weights,
    risk_parity_weights,
    volatility_target_weights,
)
from market_trader.portfolio.risk import (
    DrawdownCircuitBreaker,
    RiskLimitBreach,
    RiskLimits,
    RiskManagedStrategy,
    apply_risk_limits,
    check_order,
)
from market_trader.portfolio.weighting import (
    composite_score,
    equal_weights,
    ic_weights,
    information_coefficient,
    inverse_variance_weights,
    orthogonality_penalty,
)

__all__ = [
    "DrawdownCircuitBreaker",
    "RiskLimitBreach",
    "RiskLimits",
    "RiskManagedStrategy",
    "apply_risk_limits",
    "check_order",
    "composite_score",
    "equal_weights",
    "fractional_kelly_weights",
    "hierarchical_risk_parity",
    "ic_weights",
    "information_coefficient",
    "inverse_variance_weights",
    "ledoit_wolf_cov",
    "min_variance_weights",
    "orthogonality_penalty",
    "risk_parity_weights",
    "volatility_target_weights",
]
