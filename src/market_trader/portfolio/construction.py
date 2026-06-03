"""Portfolio construction and sizing.

Covariance via Ledoit-Wolf shrinkage; sizing via volatility targeting and
fractional Kelly; allocation via risk-parity, Hierarchical Risk Parity (López de
Prado), or minimum-variance. All operate on returns/covariance frames indexed by
symbol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance — well-conditioned even when n ≈ p."""
    clean = returns.dropna()
    estimate = LedoitWolf().fit(clean.to_numpy(dtype=float)).covariance_
    return pd.DataFrame(estimate, index=returns.columns, columns=returns.columns)


def volatility_target_weights(
    weights: pd.Series, cov: pd.DataFrame, target_vol: float, *, periods_per_year: int = 252
) -> pd.Series:
    """Scale a weight vector so its annualised portfolio volatility equals ``target_vol``."""
    w = weights.reindex(cov.index).fillna(0.0).to_numpy(dtype=float)
    period_var = float(w @ cov.to_numpy(dtype=float) @ w)
    annual_vol = np.sqrt(max(period_var, 0.0)) * np.sqrt(periods_per_year)
    if annual_vol <= 0:
        return weights
    return weights * (target_vol / annual_vol)


def fractional_kelly_weights(
    expected_returns: pd.Series, cov: pd.DataFrame, *, fraction: float = 0.25
) -> pd.Series:
    """Kelly-optimal weights ``Σ⁻¹ μ`` scaled by a (heavily) fractional multiplier."""
    inv = np.linalg.pinv(cov.to_numpy(dtype=float))
    mu = expected_returns.reindex(cov.index).fillna(0.0).to_numpy(dtype=float)
    return pd.Series(fraction * (inv @ mu), index=cov.index)


def min_variance_weights(cov: pd.DataFrame) -> pd.Series:
    """Long/short minimum-variance weights summing to 1."""
    inv = np.linalg.pinv(cov.to_numpy(dtype=float))
    ones = np.ones(len(cov))
    raw = inv @ ones
    return pd.Series(raw / raw.sum(), index=cov.index)


def risk_parity_weights(cov: pd.DataFrame, *, iters: int = 1000, tol: float = 1e-10) -> pd.Series:
    """Long-only equal-risk-contribution weights (iterative)."""
    sigma = cov.to_numpy(dtype=float)
    n = len(sigma)
    w = np.ones(n) / n
    for _ in range(iters):
        marginal = np.where((sigma @ w) <= 0, 1e-12, sigma @ w)
        w_new = np.sqrt(w / marginal)  # geometric step toward equal risk contribution
        w_new = w_new / w_new.sum()
        if np.abs(w_new - w).max() < tol:
            w = w_new
            break
        w = w_new
    return pd.Series(w, index=cov.index)


def _quasi_diagonal(link: np.ndarray) -> list[int]:
    link = link.astype(int)
    order = pd.Series([link[-1, 0], link[-1, 1]])
    n_items = link[-1, 3]
    while order.max() >= n_items:
        order.index = list(range(0, order.shape[0] * 2, 2))
        clusters = order[order >= n_items]
        i = clusters.index
        j = clusters.to_numpy() - n_items
        order[i] = link[j, 0]
        order = pd.concat([order, pd.Series(link[j, 1], index=i + 1)]).sort_index()
        order.index = list(range(order.shape[0]))
    return order.tolist()


def _cluster_variance(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items].to_numpy(dtype=float)
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hierarchical_risk_parity(returns: pd.DataFrame) -> pd.Series:
    """HRP: cluster by correlation distance, then recursively bisect by risk."""
    cols = list(returns.columns)
    if len(cols) == 1:
        return pd.Series([1.0], index=cols)
    cov = returns.cov()
    corr = returns.corr().fillna(0.0)
    dist = np.sqrt(np.clip((1.0 - corr.to_numpy(dtype=float)) / 2.0, 0.0, None))
    link = sch.linkage(squareform(dist, checks=False), method="single")
    ordered = [cols[i] for i in _quasi_diagonal(link)]

    weights = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        clusters = [
            c[start:stop]
            for c in clusters
            for start, stop in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left, var_right = _cluster_variance(cov, left), _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights.reindex(cols)
