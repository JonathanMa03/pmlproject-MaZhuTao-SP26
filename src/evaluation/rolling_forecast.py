from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    log_predictive_score_univariate,
    var_from_draws,
    es_from_draws,
    hit_var,
)


def _fit_model_instance(model_class, model_kwargs: dict[str, Any], train_df: pd.DataFrame):
    model = model_class(**model_kwargs)
    result = model.fit(train_df)
    return model, result


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

    step_size=1  -> evaluate every date
    step_size=20 -> evaluate every 20th date
    """
    cols = [c for c in data.columns if c != date_col]
    if target_col not in cols:
        raise ValueError(f"{target_col} not found in data columns.")
    if step_size < 1:
        raise ValueError("step_size must be at least 1.")

    target_idx = cols.index(target_col)
    rows = []
    n = len(data)

    eval_points = list(range(window_size, n - 1, step_size))

    for i, end_train in enumerate(eval_points):
        if verbose and (i % 10 == 0):
            print(f"[{model_class.__name__}] step {i+1}/{len(eval_points)}")

        train_df = data.iloc[end_train - window_size:end_train].copy()
        test_row = data.iloc[end_train].copy()

        model, _ = _fit_model_instance(model_class, model_kwargs, train_df)

        last_obs = train_df[cols].to_numpy(dtype=float)[-p:, :]
        sims = model.simulate_one_step(last_obs, n_sim=n_sim)
        target_draws = sims[:, target_idx]
        y_true = float(test_row[target_col])

        lps = log_predictive_score_univariate(y_true, target_draws)
        var_alpha = var_from_draws(target_draws, alpha=alpha)
        es_alpha = es_from_draws(target_draws, alpha=alpha)
        var_hit = hit_var(y_true, var_alpha)

        rows.append(
            {
                "date": test_row[date_col],
                "y_true": y_true,
                "pred_mean": float(np.mean(target_draws)),
                "pred_std": float(np.std(target_draws, ddof=1)),
                "lps": lps,
                "var_alpha": var_alpha,
                "es_alpha": es_alpha,
                "var_hit": var_hit,
            }
        )

    return pd.DataFrame(rows)


def summarize_rolling_results(results_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    return {
        "n_forecasts": int(len(results_df)),
        "mean_lps": float(results_df["lps"].mean()),
        "total_lps": float(results_df["lps"].sum()),
        "avg_pred_std": float(results_df["pred_std"].mean()),
        "var_hit_rate": float(results_df["var_hit"].mean()),
        "expected_var_rate": float(alpha),
        "mean_es": float(results_df["es_alpha"].mean()),
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
    all_results: dict[str, pd.DataFrame] = {}
    summary_rows = []

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