from __future__ import annotations

import numpy as np


def autocorrelation(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """
    Sample autocorrelation function for a 1D chain.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if max_lag is None:
        max_lag = min(n - 1, 200)

    x_centered = x - x.mean()
    var = np.dot(x_centered, x_centered) / n
    if var <= 0:
        return np.ones(max_lag + 1)

    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.dot(x_centered[:-lag], x_centered[lag:]) / (n * var)
    return acf


def effective_sample_size(x: np.ndarray, max_lag: int | None = None) -> float:
    """
    Single-chain ESS using initial positive sequence style truncation.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if n < 3:
        return float(n)

    acf = autocorrelation(x, max_lag=max_lag)

    # truncate when pair sums become negative
    rho_sum = 0.0
    for k in range(1, len(acf) - 1, 2):
        pair_sum = acf[k] + acf[k + 1]
        if pair_sum < 0:
            break
        rho_sum += pair_sum

    ess = n / (1.0 + 2.0 * rho_sum)
    return float(max(1.0, min(ess, n)))


def split_rhat(chains: np.ndarray) -> float:
    """
    Split-Rhat for multiple chains.

    Parameters
    ----------
    chains : np.ndarray
        shape (n_chains, n_draws)

    Returns
    -------
    float
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError("chains must have shape (n_chains, n_draws)")

    m, n = chains.shape
    if m < 2 or n < 4:
        raise ValueError("Need at least 2 chains and at least 4 draws per chain.")

    half = n // 2
    chains = chains[:, : 2 * half]
    split = np.concatenate([chains[:, :half], chains[:, half:]], axis=0)

    m2, n2 = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)

    W = chain_vars.mean()
    B = n2 * chain_means.var(ddof=1)
    var_hat = ((n2 - 1) / n2) * W + (1 / n2) * B

    return float(np.sqrt(var_hat / W))


def extract_scalar_chain_from_beta_draws(
    beta_draws: list[np.ndarray],
    equation_idx: int = 0,
    coef_idx: int = 0,
) -> np.ndarray:
    """
    Extract one scalar chain from a list of beta draw matrices.
    """
    return np.array([draw[equation_idx, coef_idx] for draw in beta_draws], dtype=float)


def extract_scalar_chain_from_sigma_draws(
    sigma_draws: list[np.ndarray],
    i: int = 0,
    j: int = 0,
    component: int | None = None,
) -> np.ndarray:
    """
    Extract one scalar chain from sigma draws.

    For Gaussian / Student-t:
        sigma_draws[s] is (d, d)

    For Mixture:
        sigma_draws[s] is (K, d, d)
        then component must be provided.
    """
    out = []
    for draw in sigma_draws:
        arr = np.asarray(draw)
        if arr.ndim == 2:
            out.append(arr[i, j])
        elif arr.ndim == 3:
            if component is None:
                raise ValueError("For mixture sigma draws, provide component index.")
            out.append(arr[component, i, j])
        else:
            raise ValueError("Unexpected sigma draw dimension.")
    return np.array(out, dtype=float)