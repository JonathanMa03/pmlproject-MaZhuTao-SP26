from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_var import BaseVAR, VARDesign


@dataclass
class GaussianVARResult:
    """
    Container for fitted Gaussian VAR results.

    Attributes
    ----------
    beta : np.ndarray
        Coefficient matrix of shape (d, k), where each row is one equation.
    sigma : np.ndarray
        Residual covariance matrix of shape (d, d).
    fitted_values : np.ndarray
        In-sample fitted values of shape (T-p, d).
    residuals : np.ndarray
        In-sample residuals of shape (T-p, d).
    design : VARDesign
        Design matrices and metadata used in fitting.
    """
    beta: np.ndarray
    sigma: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    design: VARDesign


class GaussianVAR:
    """
    Gaussian VAR estimated equation-by-equation by OLS.

    Model
    -----
        y_t = B x_t + eps_t
        eps_t ~ N(0, Sigma)

    where x_t contains lagged values and optionally an intercept.
    """

    def __init__(self, p: int = 1, intercept: bool = True):
        self.p = p
        self.intercept = intercept
        self.base_var = BaseVAR(p=p, intercept=intercept)
        self.result_: Optional[GaussianVARResult] = None

    def fit(self, data: pd.DataFrame, date_col: str = "date") -> GaussianVARResult:
        """
        Fit Gaussian VAR by ordinary least squares.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with one date column and the remaining columns as numeric series.
        date_col : str
            Name of the date column.

        Returns
        -------
        GaussianVARResult
            Fitted model result object.
        """
        design = self.base_var.build_design(data, date_col=date_col)
        X, Y = design.X, design.Y

        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")
        if not np.isfinite(Y).all():
            raise ValueError("Y contains non-finite values.")

        X = np.asarray(X, dtype=np.float64, order="C")
        Y = np.asarray(Y, dtype=np.float64, order="C")

        # OLS using least squares (numerically safer than explicit normal equations)
        beta_t, residuals_lstsq, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        beta_t = np.asarray(beta_t, dtype=np.float64, order="C")

        if not np.isfinite(beta_t).all():
            raise ValueError("Estimated coefficients contain non-finite values.")

        # beta_t has shape (k, d); store beta as (d, k)
        beta = beta_t.T

        # Use einsum to avoid possible BLAS/matmul warnings on some systems
        fitted_values = np.einsum("ij,jk->ik", X, beta_t, optimize=True)
        residuals = Y - fitted_values

        if not np.isfinite(fitted_values).all():
            raise ValueError("Fitted values contain non-finite values.")
        if not np.isfinite(residuals).all():
            raise ValueError("Residuals contain non-finite values.")

        T_eff, d = residuals.shape
        k = X.shape[1]

        if T_eff <= k:
            raise ValueError(
                f"Not enough effective observations to estimate covariance: "
                f"T_eff={T_eff}, k={k}."
            )

        sigma = (residuals.T @ residuals) / (T_eff - k)

        if not np.isfinite(sigma).all():
            raise ValueError("Residual covariance matrix contains non-finite values.")

        self.result_ = GaussianVARResult(
            beta=beta,
            sigma=sigma,
            fitted_values=fitted_values,
            residuals=residuals,
            design=design,
        )
        return self.result_

    def predict_in_sample(self) -> np.ndarray:
        """
        Return fitted values for the training sample.
        """
        self._check_is_fitted()
        return self.result_.fitted_values

    def get_residuals(self) -> np.ndarray:
        """
        Return in-sample residuals.
        """
        self._check_is_fitted()
        return self.result_.residuals

    def forecast_one_step(self, last_observations: np.ndarray) -> np.ndarray:
        """
        Compute one-step-ahead forecast mean.

        Parameters
        ----------
        last_observations : np.ndarray
            Array of shape (p, d), ordered from oldest to newest.

        Returns
        -------
        np.ndarray
            Forecast mean vector of shape (d,).
        """
        self._check_is_fitted()

        if last_observations.ndim != 2:
            raise ValueError("last_observations must have shape (p, d).")

        p, d = last_observations.shape
        if p != self.p:
            raise ValueError(f"Expected {self.p} lag rows, got {p}.")
        if d != self.result_.design.d:
            raise ValueError(f"Expected d={self.result_.design.d}, got {d}.")

        x = []

        if self.intercept:
            x.append(1.0)

        # Build x_t in the same order as BaseVAR: lag1, lag2, ..., lagp
        for lag in range(1, self.p + 1):
            x.extend(last_observations[-lag, :].tolist())

        x = np.asarray(x, dtype=np.float64)
        return self.result_.beta @ x

    def simulate_one_step(self, last_observations: np.ndarray, n_sim: int = 1000) -> np.ndarray:
        """
        Simulate one-step-ahead draws from the Gaussian predictive distribution.

        Parameters
        ----------
        last_observations : np.ndarray
            Array of shape (p, d), ordered from oldest to newest.
        n_sim : int
            Number of simulations.

        Returns
        -------
        np.ndarray
            Simulated draws of shape (n_sim, d).
        """
        self._check_is_fitted()

        mean = self.forecast_one_step(last_observations)
        sims = np.random.multivariate_normal(mean=mean, cov=self.result_.sigma, size=n_sim)
        return sims

    def summary(self) -> dict:
        """
        Return a compact summary dictionary.
        """
        self._check_is_fitted()

        return {
            "p": self.p,
            "intercept": self.intercept,
            "n_obs_effective": self.result_.design.Y.shape[0],
            "n_series": self.result_.design.d,
            "n_features": self.result_.design.X.shape[1],
            "beta_shape": self.result_.beta.shape,
            "sigma_shape": self.result_.sigma.shape,
        }

    def _check_is_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit(...) first.")