from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_var import BaseVAR, VARDesign


@dataclass
class StudentTVARResult:
    beta_draws: list[np.ndarray]
    sigma_draws: list[np.ndarray]
    lambda_draws: list[np.ndarray]
    fitted_values_last: np.ndarray
    residuals_last: np.ndarray
    design: VARDesign
    nu: float


class StudentTVAR:
    """
    Bayesian VAR with Student-t innovations using scale-mixture augmentation.

    eps_t | lambda_t ~ N(0, Sigma / lambda_t)
    lambda_t ~ Gamma(nu/2, nu/2)

    This implementation avoids @ matmul in critical places because some
    local BLAS/Accelerate setups can emit spurious overflow/divide warnings
    even when the arrays are finite and well-scaled.
    """

    def __init__(
        self,
        p: int = 1,
        intercept: bool = True,
        nu: float = 8.0,
        seed: int = 123,
        ridge_beta: float = 1e-6,
        ridge_sigma: float = 1e-6,
        lambda_min: float = 1e-4,
        lambda_max: float = 1e4,
    ):
        if nu <= 2:
            raise ValueError("nu must be > 2.")
        self.p = p
        self.intercept = intercept
        self.nu = float(nu)
        self.rng = np.random.default_rng(seed)
        self.base_var = BaseVAR(p=p, intercept=intercept)

        self.ridge_beta = ridge_beta
        self.ridge_sigma = ridge_sigma
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

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

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")
        if not np.isfinite(Y).all():
            raise ValueError("Y contains non-finite values.")

        T_eff, d = Y.shape
        k = X.shape[1]

        # OLS initialization using lstsq
        beta_t, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (k, d)
        beta = beta_t.T                                       # (d, k)

        fitted_values = np.einsum("tk,kd->td", X, beta_t)
        residuals = Y - fitted_values

        Sigma = np.einsum("ti,tj->ij", residuals, residuals) / max(T_eff - k, 1)
        Sigma = Sigma + self.ridge_sigma * np.eye(d)

        lam = np.ones(T_eff, dtype=np.float64)

        beta_draws: list[np.ndarray] = []
        sigma_draws: list[np.ndarray] = []
        lambda_draws: list[np.ndarray] = []

        fitted_values_last = None
        residuals_last = None

        for it in range(n_iter):
            # ---------------------------------------
            # 1) beta | Sigma, lambda, Y
            # weighted least squares with ridge
            # ---------------------------------------
            W_sqrt = np.sqrt(lam)[:, None]
            Xw = X * W_sqrt
            Yw = Y * W_sqrt

            XtX = np.einsum("ti,tj->ij", Xw, Xw) + self.ridge_beta * np.eye(k)
            XtY = np.einsum("ti,td->id", Xw, Yw)

            if not np.isfinite(XtX).all():
                raise RuntimeError(f"XtX became non-finite at iteration {it}.")
            if not np.isfinite(XtY).all():
                raise RuntimeError(f"XtY became non-finite at iteration {it}.")

            beta_t = np.linalg.solve(XtX, XtY)   # (k, d)
            beta = beta_t.T                      # (d, k)

            fitted_values = np.einsum("tk,kd->td", X, beta_t)
            residuals = Y - fitted_values

            if not np.isfinite(fitted_values).all():
                raise RuntimeError(f"Non-finite fitted values at iteration {it}.")
            if not np.isfinite(residuals).all():
                raise RuntimeError(f"Non-finite residuals at iteration {it}.")

            # ---------------------------------------
            # 2) lambda_t | beta, Sigma, Y
            # ---------------------------------------
            Sigma_inv = np.linalg.inv(Sigma)

            for t in range(T_eff):
                e_t = residuals[t]
                quad = float(np.einsum("i,ij,j->", e_t, Sigma_inv, e_t))

                shape = 0.5 * (self.nu + d)
                rate = 0.5 * (self.nu + quad)

                lam[t] = self.rng.gamma(shape=shape, scale=1.0 / rate)

            lam = np.clip(lam, self.lambda_min, self.lambda_max)

            # ---------------------------------------
            # 3) Sigma | beta, lambda, Y
            # ---------------------------------------
            Ew = residuals * np.sqrt(lam)[:, None]
            Sigma = np.einsum("ti,tj->ij", Ew, Ew) / T_eff
            Sigma = Sigma + self.ridge_sigma * np.eye(d)

            if not np.isfinite(Sigma).all():
                raise RuntimeError(f"Non-finite Sigma at iteration {it}.")

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
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")

        beta = self.posterior_mean_beta()

        p, d = last_observations.shape
        if p != self.p:
            raise ValueError(f"Expected {self.p} lags, got {p}.")
        if d != self.result_.design.d:
            raise ValueError(f"Expected {self.result_.design.d} series, got {d}.")

        x = []
        if self.intercept:
            x.append(1.0)
        for lag in range(1, self.p + 1):
            x.extend(last_observations[-lag, :].tolist())

        x = np.asarray(x, dtype=np.float64)
        return np.einsum("dk,k->d", beta, x)

    def simulate_one_step(self, last_observations: np.ndarray, n_sim: int = 1000) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")

        p, d = last_observations.shape
        if p != self.p:
            raise ValueError(f"Expected {self.p} lags, got {p}.")
        if d != self.result_.design.d:
            raise ValueError(f"Expected {self.result_.design.d} series, got {d}.")

        x = []
        if self.intercept:
            x.append(1.0)
        for lag in range(1, self.p + 1):
            x.extend(last_observations[-lag, :].tolist())
        x = np.asarray(x, dtype=np.float64)

        sims = np.zeros((n_sim, d))
        n_draws = len(self.result_.beta_draws)

        for i in range(n_sim):
            s = self.rng.integers(0, n_draws)
            beta = self.result_.beta_draws[s]
            Sigma = self.result_.sigma_draws[s]

            mean = np.einsum("dk,k->d", beta, x)
            lam_next = self.rng.gamma(shape=self.nu / 2.0, scale=2.0 / self.nu)
            lam_next = np.clip(lam_next, self.lambda_min, self.lambda_max)

            sims[i] = self.rng.multivariate_normal(mean, Sigma / lam_next)

        return sims