from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_var import BaseVAR, VARDesign


@dataclass
class StudentTVARResult:
    beta_draws: list[np.ndarray]       # each draw shape (d, k)
    sigma_draws: list[np.ndarray]      # each draw shape (d, d)
    lambda_draws: list[np.ndarray]     # each draw shape (T_eff,)
    fitted_values_last: np.ndarray     # shape (T_eff, d)
    residuals_last: np.ndarray         # shape (T_eff, d)
    design: VARDesign
    nu: float


class StudentTVAR:
    """
    Bayesian VAR with Student-t innovations using scale-mixture augmentation.

    Model
    -----
    y_t = B x_t + eps_t
    eps_t | lambda_t ~ N(0, Sigma / lambda_t)
    lambda_t ~ Gamma(nu/2, nu/2)

    Notes
    -----
    - Uses Gibbs sampling with latent scales lambda_t.
    - Uses weighted least squares for beta updates.
    - Uses a simple inverse-Wishart-style covariance update approximation.
    """

    def __init__(
        self,
        p: int = 1,
        intercept: bool = True,
        nu: float = 8.0,
        seed: int = 123,
    ):
        if nu <= 2:
            raise ValueError("nu must be > 2 for finite variance.")
        self.p = p
        self.intercept = intercept
        self.nu = float(nu)
        self.base_var = BaseVAR(p=p, intercept=intercept)
        self.rng = np.random.default_rng(seed)
        self.result_: Optional[StudentTVARResult] = None

    def fit(
        self,
        data: pd.DataFrame,
        date_col: str = "date",
        n_iter: int = 1500,
        burn: int = 500,
    ) -> StudentTVARResult:
        design = self.base_var.build_design(data, date_col=date_col)
        X, Y = design.X, design.Y

        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")
        if not np.isfinite(Y).all():
            raise ValueError("Y contains non-finite values.")

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        T_eff, d = Y.shape
        k = X.shape[1]

        # Initialize
        beta_t, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (k, d)
        beta = beta_t.T                                       # (d, k)

        residuals = Y - X @ beta_t
        Sigma = (residuals.T @ residuals) / max(T_eff - k, 1)
        lam = np.ones(T_eff, dtype=np.float64)

        beta_draws: list[np.ndarray] = []
        sigma_draws: list[np.ndarray] = []
        lambda_draws: list[np.ndarray] = []

        fitted_values_last = None
        residuals_last = None

        for it in range(n_iter):
            # -------------------------------------------------
            # 1) beta | Sigma, lambda, Y
            #    Weighted least squares with weights lambda_t
            # -------------------------------------------------
            W_sqrt = np.sqrt(lam)[:, None]      # (T_eff, 1)
            Xw = X * W_sqrt                     # (T_eff, k)
            Yw = Y * W_sqrt                     # (T_eff, d)

            beta_t, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)  # (k, d)
            beta = beta_t.T                                        # (d, k)

            fitted_values = X @ beta_t
            residuals = Y - fitted_values

            # -------------------------------------------------
            # 2) lambda_t | beta, Sigma, Y
            #    Gamma update from Student-t mixture form
            # -------------------------------------------------
            Sigma_inv = np.linalg.inv(Sigma)
            for t in range(T_eff):
                e_t = residuals[t]
                quad = float(e_t.T @ Sigma_inv @ e_t)

                shape = 0.5 * (self.nu + d)
                rate = 0.5 * (self.nu + quad)

                # numpy gamma uses shape and scale = 1/rate
                lam[t] = self.rng.gamma(shape=shape, scale=1.0 / rate)

            # -------------------------------------------------
            # 3) Sigma | beta, lambda, Y
            #    Weighted residual covariance
            # -------------------------------------------------
            Ew = residuals * np.sqrt(lam)[:, None]
            Sigma = (Ew.T @ Ew) / T_eff

            # small ridge for stability
            Sigma = Sigma + 1e-8 * np.eye(d)

            fitted_values_last = fitted_values
            residuals_last = residuals

            if it >= burn:
                beta_draws.append(beta.copy())
                sigma_draws.append(Sigma.copy())
                lambda_draws.append(lam.copy())

        self.result_ = StudentTVARResult(
            beta_draws=beta_draws,
            sigma_draws=sigma_draws,
            lambda_draws=lambda_draws,
            fitted_values_last=fitted_values_last,
            residuals_last=residuals_last,
            design=design,
            nu=self.nu,
        )
        return self.result_

    def summary(self) -> dict:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return {
            "p": self.p,
            "intercept": self.intercept,
            "nu": self.nu,
            "n_saved_draws": len(self.result_.beta_draws),
            "n_obs_effective": self.result_.design.Y.shape[0],
            "n_series": self.result_.design.d,
            "n_features": self.result_.design.X.shape[1],
            "beta_draw_shape": self.result_.beta_draws[-1].shape,
            "sigma_draw_shape": self.result_.sigma_draws[-1].shape,
        }

    def posterior_mean_beta(self) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return np.mean(np.stack(self.result_.beta_draws, axis=0), axis=0)

    def posterior_mean_sigma(self) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return np.mean(np.stack(self.result_.sigma_draws, axis=0), axis=0)

    def forecast_one_step_mean(self, last_observations: np.ndarray) -> np.ndarray:
        """
        Posterior mean one-step-ahead forecast.
        """
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")

        beta = self.posterior_mean_beta()

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

        for lag in range(1, self.p + 1):
            x.extend(last_observations[-lag, :].tolist())

        x = np.asarray(x, dtype=np.float64)
        return beta @ x

    def simulate_one_step(self, last_observations: np.ndarray, n_sim: int = 1000) -> np.ndarray:
        """
        Simulate one-step-ahead draws from the posterior predictive distribution.
        """
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")

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
        for lag in range(1, self.p + 1):
            x.extend(last_observations[-lag, :].tolist())
        x = np.asarray(x, dtype=np.float64)

        n_draws = len(self.result_.beta_draws)
        sims = np.zeros((n_sim, d), dtype=np.float64)

        for i in range(n_sim):
            s = self.rng.integers(0, n_draws)
            beta = self.result_.beta_draws[s]
            Sigma = self.result_.sigma_draws[s]

            mean = beta @ x
            lam_next = self.rng.gamma(shape=self.nu / 2.0, scale=2.0 / self.nu)
            sims[i] = self.rng.multivariate_normal(mean=mean, cov=Sigma / lam_next)

        return sims