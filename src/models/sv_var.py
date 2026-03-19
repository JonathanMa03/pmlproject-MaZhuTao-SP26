from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_var import BaseVAR, VARDesign


@dataclass
class SVVARResult:
    beta_draws: list[np.ndarray]          # each (d, k_features)
    h_draws: list[np.ndarray]             # each (T_eff, d)
    fitted_values_last: np.ndarray        # (T_eff, d)
    residuals_last: np.ndarray            # (T_eff, d)
    design: VARDesign
    phi: float
    sigma_h: float


class SVVAR:
    """
    Bayesian VAR with diagonal stochastic volatility.

    Model
    -----
    y_t = B x_t + eps_t
    eps_t ~ N(0, diag(exp(h_t)))

    h_{t,j} = phi * h_{t-1,j} + eta_{t,j},
    eta_{t,j} ~ N(0, sigma_h^2)

    Notes
    -----
    - This is a practical diagonal-SV implementation.
    - beta updated by weighted least squares.
    - latent log-volatilities h updated by random-walk Metropolis steps.
    - good as a project model before attempting a full multivariate SV sampler.
    """

    def __init__(
        self,
        p: int = 1,
        intercept: bool = True,
        phi: float = 0.98,
        sigma_h: float = 0.15,
        seed: int = 123,
        ridge_beta: float = 1e-6,
        proposal_sd: float = 0.20,
    ):
        if not (-1 < phi < 1):
            raise ValueError("phi must lie in (-1, 1).")
        if sigma_h <= 0:
            raise ValueError("sigma_h must be positive.")
        if proposal_sd <= 0:
            raise ValueError("proposal_sd must be positive.")

        self.p = p
        self.intercept = intercept
        self.phi = float(phi)
        self.sigma_h = float(sigma_h)
        self.rng = np.random.default_rng(seed)
        self.base_var = BaseVAR(p=p, intercept=intercept)

        self.ridge_beta = ridge_beta
        self.proposal_sd = proposal_sd

        self.result_: Optional[SVVARResult] = None

    def fit(
        self,
        data: pd.DataFrame,
        date_col: str = "date",
        n_iter: int = 1500,
        burn: int = 500,
    ) -> SVVARResult:
        design = self.base_var.build_design(data, date_col=date_col)
        X, Y = design.X, design.Y

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")
        if not np.isfinite(Y).all():
            raise ValueError("Y contains non-finite values.")

        T_eff, d = Y.shape
        k_features = X.shape[1]

        # OLS initialization
        beta_t, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        beta = beta_t.T

        fitted_values = np.einsum("tk,kd->td", X, beta_t)
        residuals = Y - fitted_values

        # initialize log volatilities from residual variance
        init_var = np.var(residuals, axis=0, ddof=1)
        init_var = np.where(init_var <= 1e-8, 1e-8, init_var)
        h = np.log(init_var)[None, :] * np.ones((T_eff, d), dtype=np.float64)

        beta_draws: list[np.ndarray] = []
        h_draws: list[np.ndarray] = []

        fitted_values_last = None
        residuals_last = None

        for it in range(n_iter):
            # ---------------------------------------
            # 1) beta | h, Y
            # weighted least squares equation by equation
            # ---------------------------------------
            beta_new = np.zeros((d, k_features), dtype=np.float64)

            for j in range(d):
                sigma2_t = np.exp(h[:, j])
                weights = 1.0 / np.maximum(sigma2_t, 1e-8)

                W_sqrt = np.sqrt(weights)[:, None]
                Xw = X * W_sqrt
                yw = Y[:, j] * np.sqrt(weights)

                XtX = np.einsum("ti,tj->ij", Xw, Xw) + self.ridge_beta * np.eye(k_features)
                XtY = np.einsum("ti,t->i", Xw, yw)

                beta_new[j] = np.linalg.solve(XtX, XtY)

            beta = beta_new
            beta_t = beta.T

            fitted_values = np.einsum("tk,kd->td", X, beta_t)
            residuals = Y - fitted_values

            if not np.isfinite(residuals).all():
                raise RuntimeError(f"Non-finite residuals at iteration {it}.")

            # ---------------------------------------
            # 2) h | beta, Y
            # random-walk Metropolis update for each h_{t,j}
            # ---------------------------------------
            for j in range(d):
                for t in range(T_eff):
                    current = h[t, j]
                    proposal = current + self.rng.normal(scale=self.proposal_sd)

                    logpost_current = self._log_h_conditional(
                        h_value=current,
                        residual=residuals[t, j],
                        h_prev=h[t - 1, j] if t > 0 else None,
                        h_next=h[t + 1, j] if t < T_eff - 1 else None,
                    )

                    logpost_proposal = self._log_h_conditional(
                        h_value=proposal,
                        residual=residuals[t, j],
                        h_prev=h[t - 1, j] if t > 0 else None,
                        h_next=h[t + 1, j] if t < T_eff - 1 else None,
                    )

                    log_accept = logpost_proposal - logpost_current

                    if np.log(self.rng.uniform()) < log_accept:
                        h[t, j] = proposal

            fitted_values_last = fitted_values
            residuals_last = residuals

            if it >= burn:
                beta_draws.append(beta.copy())
                h_draws.append(h.copy())

        self.result_ = SVVARResult(
            beta_draws=beta_draws,
            h_draws=h_draws,
            fitted_values_last=fitted_values_last,
            residuals_last=residuals_last,
            design=design,
            phi=self.phi,
            sigma_h=self.sigma_h,
        )
        return self.result_

    def _log_h_conditional(
        self,
        h_value: float,
        residual: float,
        h_prev: float | None,
        h_next: float | None,
    ) -> float:
        """
        Log conditional density contribution for one latent log-volatility state.
        """
        # Observation likelihood:
        # e_t | h_t ~ N(0, exp(h_t))
        obs = -0.5 * (np.log(2.0 * np.pi) + h_value + (residual**2) / np.exp(h_value))

        prior = 0.0

        # prior from previous transition
        if h_prev is None:
            # stationary-ish initialization around 0
            prior += -0.5 * (h_value / 1.0) ** 2
        else:
            mean_prev = self.phi * h_prev
            prior += -0.5 * ((h_value - mean_prev) / self.sigma_h) ** 2

        # prior from next transition
        if h_next is not None:
            mean_next = self.phi * h_value
            prior += -0.5 * ((h_next - mean_next) / self.sigma_h) ** 2

        return obs + prior

    def summary(self) -> dict:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return {
            "p": self.p,
            "intercept": self.intercept,
            "phi": self.phi,
            "sigma_h": self.sigma_h,
            "n_saved_draws": len(self.result_.beta_draws),
            "n_obs_effective": self.result_.design.Y.shape[0],
            "n_series": self.result_.design.d,
            "n_features": self.result_.design.X.shape[1],
            "beta_draw_shape": self.result_.beta_draws[-1].shape,
            "h_draw_shape": self.result_.h_draws[-1].shape,
        }

    def posterior_mean_beta(self) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return np.mean(np.stack(self.result_.beta_draws, axis=0), axis=0)

    def posterior_mean_h(self) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return np.mean(np.stack(self.result_.h_draws, axis=0), axis=0)

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
            h_last = self.result_.h_draws[s][-1]   # last latent vol state

            mean = np.einsum("dk,k->d", beta, x)

            # propagate log-vol one step ahead
            h_next = self.phi * h_last + self.rng.normal(scale=self.sigma_h, size=d)
            Sigma_next = np.diag(np.exp(h_next))

            sims[i] = self.rng.multivariate_normal(mean, Sigma_next)

        return sims