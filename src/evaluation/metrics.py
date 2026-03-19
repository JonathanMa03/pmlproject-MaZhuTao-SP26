from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def log_predictive_score_from_draws(
    y_true: np.ndarray,
    predictive_draws: np.ndarray,
    jitter: float = 1e-6,
) -> float:
    """
    Approximate log predictive density at y_true using KDE on predictive draws.

    Parameters
    ----------
    y_true : np.ndarray
        True realized vector, shape (d,)
    predictive_draws : np.ndarray
        Simulated predictive draws, shape (n_sim, d)
    jitter : float
        Small jitter added to avoid singular KDE issues.

    Returns
    -------
    float
        Approximate log predictive score
    """
    y_true = np.asarray(y_true, dtype=float)
    predictive_draws = np.asarray(predictive_draws, dtype=float)

    if predictive_draws.ndim != 2:
        raise ValueError("predictive_draws must be 2D: (n_sim, d)")

    n_sim, d = predictive_draws.shape
    if y_true.shape[0] != d:
        raise ValueError("y_true dimension does not match predictive_draws")

    # KDE on multivariate draws
    vals = predictive_draws.T
    vals = vals + jitter * np.random.randn(*vals.shape)
    kde = gaussian_kde(vals)
    density = float(kde(y_true))
    density = max(density, 1e-300)
    return np.log(density)


def log_predictive_score_univariate(
    y_true: float,
    predictive_draws: np.ndarray,
    jitter: float = 1e-6,
) -> float:
    """
    Univariate log predictive score using KDE.
    """
    predictive_draws = np.asarray(predictive_draws, dtype=float).ravel()
    vals = predictive_draws + jitter * np.random.randn(len(predictive_draws))
    kde = gaussian_kde(vals)
    density = float(kde(y_true))
    density = max(density, 1e-300)
    return np.log(density)


def var_from_draws(draws: np.ndarray, alpha: float = 0.05) -> float:
    """
    Value-at-Risk from predictive draws for a univariate return series.
    """
    draws = np.asarray(draws, dtype=float).ravel()
    return float(np.quantile(draws, alpha))


def es_from_draws(draws: np.ndarray, alpha: float = 0.05) -> float:
    """
    Expected Shortfall from predictive draws for a univariate return series.
    """
    draws = np.asarray(draws, dtype=float).ravel()
    q = np.quantile(draws, alpha)
    tail = draws[draws <= q]
    if len(tail) == 0:
        return float(q)
    return float(tail.mean())


def hit_var(y_true: float, var_alpha: float) -> int:
    """
    Indicator for VaR violation.
    """
    return int(y_true <= var_alpha)


def predictive_summary(draws: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Return summary stats from predictive draws for one univariate series.
    """
    draws = np.asarray(draws, dtype=float).ravel()
    return {
        "mean": float(np.mean(draws)),
        "std": float(np.std(draws, ddof=1)),
        "var_alpha": var_from_draws(draws, alpha=alpha),
        "es_alpha": es_from_draws(draws, alpha=alpha),
    }