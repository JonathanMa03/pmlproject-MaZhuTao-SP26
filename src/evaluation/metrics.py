from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def _safe_kde_logpdf(
    y_true: np.ndarray | float,
    draws: np.ndarray,
    jitter: float = 1e-8,
) -> float:
    """
    Safely evaluate log density from simulated predictive draws using KDE.
    """
    draws = np.asarray(draws, dtype=float)

    if draws.ndim == 1:
        vals = draws + jitter * np.random.randn(len(draws))
        kde = gaussian_kde(vals)
        density = float(kde(y_true))
    elif draws.ndim == 2:
        vals = draws.T + jitter * np.random.randn(*draws.T.shape)
        kde = gaussian_kde(vals)
        density = float(kde(np.asarray(y_true, dtype=float)))
    else:
        raise ValueError("draws must be 1D or 2D.")

    density = max(density, 1e-300)
    return float(np.log(density))


def log_predictive_score_from_draws(
    y_true: np.ndarray,
    predictive_draws: np.ndarray,
    jitter: float = 1e-8,
) -> float:
    """
    Multivariate log predictive score from simulated predictive draws.

    Parameters
    ----------
    y_true : np.ndarray
        Realized vector, shape (d,)
    predictive_draws : np.ndarray
        Simulated predictive draws, shape (n_sim, d)
    jitter : float
        Small jitter to stabilize KDE.

    Returns
    -------
    float
        Approximate log predictive score.
    """
    y_true = np.asarray(y_true, dtype=float)
    predictive_draws = np.asarray(predictive_draws, dtype=float)

    if predictive_draws.ndim != 2:
        raise ValueError("predictive_draws must be 2D: (n_sim, d)")

    _, d = predictive_draws.shape
    if y_true.shape[0] != d:
        raise ValueError("y_true dimension does not match predictive_draws")

    return _safe_kde_logpdf(y_true, predictive_draws, jitter=jitter)


def log_predictive_score_univariate(
    y_true: float,
    predictive_draws: np.ndarray,
    jitter: float = 1e-8,
) -> float:
    """
    Univariate log predictive score from simulated predictive draws.
    """
    predictive_draws = np.asarray(predictive_draws, dtype=float).ravel()
    return _safe_kde_logpdf(float(y_true), predictive_draws, jitter=jitter)


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
    Indicator for a single VaR violation.
    Returns 1 if the realized value falls below the VaR threshold, else 0.
    """
    return int(float(y_true) <= float(var_alpha))


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


def one_step_metrics(
    y_true: float,
    draws: np.ndarray,
    alpha: float = 0.05,
    jitter: float = 1e-8,
) -> dict:
    """
    One-step predictive metrics for a single realized observation.

    Notes
    -----
    'var_hit' is a single 0/1 indicator for one forecast origin.
    A meaningful VaR hit *rate* must be computed across many rolling forecasts.
    """
    draws = np.asarray(draws, dtype=float).ravel()

    var_alpha = var_from_draws(draws, alpha=alpha)
    es_alpha = es_from_draws(draws, alpha=alpha)

    return {
        "lps": log_predictive_score_univariate(y_true, draws, jitter=jitter),
        "VaR_5%": var_alpha,
        "ES_5%": es_alpha,
        "var_hit": hit_var(y_true, var_alpha),
    }