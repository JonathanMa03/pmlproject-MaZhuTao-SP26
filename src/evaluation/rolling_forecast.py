from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import one_step_metrics


def _fit_model_instance(model_class, model_kwargs: dict[str, Any], train_df: pd.DataFrame):
    model = model_class(**model_kwargs)
    model.fit(train_df)
    return model


def _simulate_one_step_generic(
    model,
    last_obs: np.ndarray,
    n_sim: int,
) -> np.ndarray:
    """
    Dispatch one-step simulation.

    Notes
    -----
    - For most models, we use model.simulate_one_step(last_obs, n_sim=n_sim).
    - For SV models, this still works if sv_var.py implements proper volatility propagation
      inside simulate_one_step().
    """
    if not hasattr(model, "simulate_one_step"):
        raise AttributeError(
            f"{type(model).__name__} does not implement simulate_one_step()."
        )

    sims = model.simulate_one_step(last_obs, n_sim=n_sim)
    sims = np.asarray(sims, dtype=float)

    if sims.ndim != 2:
        raise ValueError(
            f"simulate_one_step() for {type(model).__name__} must return shape (n_sim, d). "
            f"Got shape {sims.shape}."
        )
    return sims


def rolling_forecast_univariate_target(
    data: pd.DataFrame,
    model_class,
    model_kwargs: dict[str, Any],
    target_col: str,
    window_size: int,
    p: int,
    n_sim: int = 1000,
    alpha: float = 0.05,
    date_col: str = "date",
    step_size: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Rolling one-step-ahead forecast evaluation for one target series.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing a date column and modeled series.
    model_class :
        Model class to fit at each rolling step.
    model_kwargs : dict[str, Any]
        Keyword arguments for the model class.
    target_col : str
        Target series to evaluate.
    window_size : int
        Number of training observations in each rolling window.
    p : int
        VAR lag order.
    n_sim : int
        Number of predictive simulations per forecast origin.
    alpha : float
        Tail probability for VaR / ES.
    date_col : str
        Name of date column.
    step_size : int
        Evaluate every `step_size`-th forecast origin.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        One row per rolling forecast origin.
    """
    cols = [c for c in data.columns if c != date_col]

    if target_col not in cols:
        raise ValueError(f"{target_col} not found in data columns.")
    if step_size < 1:
        raise ValueError("step_size must be at least 1.")
    if window_size <= p:
        raise ValueError("window_size must be strictly larger than p.")

    target_idx = cols.index(target_col)
    rows: list[dict[str, Any]] = []
    n = len(data)

    eval_points = list(range(window_size, n, step_size))

    for i, end_train in enumerate(eval_points):
        if verbose and (i % 10 == 0):
            print(f"[{model_class.__name__}] step {i + 1}/{len(eval_points)}")

        train_df = data.iloc[end_train - window_size : end_train].copy()
        test_row = data.iloc[end_train].copy()

        model = _fit_model_instance(model_class, model_kwargs, train_df)

        last_obs = train_df[cols].to_numpy(dtype=float)[-p:, :]
        sims = _simulate_one_step_generic(model, last_obs, n_sim=n_sim)
        target_draws = sims[:, target_idx]
        y_true = float(test_row[target_col])

        metrics = one_step_metrics(y_true, target_draws, alpha=alpha)

        rows.append(
            {
                "date": test_row[date_col],
                "y_true": y_true,
                "pred_mean": float(np.mean(target_draws)),
                "pred_std": float(np.std(target_draws, ddof=1)),
                "lps": metrics["lps"],
                "VaR_5%": metrics["VaR_5%"],
                "ES_5%": metrics["ES_5%"],
                "var_hit": metrics["var_hit"],
            }
        )

    return pd.DataFrame(rows)


def summarize_rolling_results(results_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Summarize rolling forecast results.

    Notes
    -----
    - var_hit_rate is meaningful only across many forecast origins.
    """
    return {
        "n_forecasts": int(len(results_df)),
        "mean_lps": float(results_df["lps"].mean()),
        "total_lps": float(results_df["lps"].sum()),
        "avg_pred_std": float(results_df["pred_std"].mean()),
        "var_hit_rate": float(results_df["var_hit"].mean()),
        "expected_var_rate": float(alpha),
        "mean_VaR_5%": float(results_df["VaR_5%"].mean()),
        "mean_ES_5%": float(results_df["ES_5%"].mean()),
    }


def compare_models_rolling(
    data: pd.DataFrame,
    model_specs: dict[str, tuple[Any, dict[str, Any]]],
    target_col: str,
    window_size: int,
    p: int,
    n_sim: int = 1000,
    alpha: float = 0.05,
    date_col: str = "date",
    step_size: int = 1,
    verbose: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Compare multiple models on rolling one-step-ahead forecasts.

    Returns
    -------
    all_results : dict[str, pd.DataFrame]
        Per-model rolling forecast outputs.
    summary_df : pd.DataFrame
        Summary comparison table.
    """
    all_results: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for name, (model_class, model_kwargs) in model_specs.items():
        if verbose:
            print(f"\nRunning model: {name}")

        res = rolling_forecast_univariate_target(
            data=data,
            model_class=model_class,
            model_kwargs=model_kwargs,
            target_col=target_col,
            window_size=window_size,
            p=p,
            n_sim=n_sim,
            alpha=alpha,
            date_col=date_col,
            step_size=step_size,
            verbose=verbose,
        )
        all_results[name] = res

        summ = summarize_rolling_results(res, alpha=alpha)
        summ["model"] = name
        summary_rows.append(summ)

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("mean_lps", ascending=False)
        .reset_index(drop=True)
    )
    return all_results, summary_df