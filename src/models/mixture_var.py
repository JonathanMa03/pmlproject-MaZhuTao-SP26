from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_var import BaseVAR, VARDesign


@dataclass
class MixtureVARResult:
    beta_draws: list[np.ndarray]          # each (d, k_features)
    sigma_draws: list[np.ndarray]         # each (K, d, d)
    pi_draws: list[np.ndarray]            # each (K,)
    z_draws: list[np.ndarray]             # each (T_eff,)
    fitted_values_last: np.ndarray        # (T_eff, d)
    residuals_last: np.ndarray            # (T_eff, d)
    design: VARDesign
    n_components: int


class MixtureVAR:
    """
    Bayesian VAR with Gaussian mixture innovations.

    Model
    -----
    y_t = B x_t + eps_t
    eps_t | z_t=k ~ N(0, Sigma_k)
    z_t ~ Categorical(pi)

    Notes
    -----
    - beta is shared across regimes
    - each regime k has its own covariance Sigma_k
    - Gibbs-style updates for z, pi, Sigma_k
    - beta updated by weighted least squares using regime-specific precision proxy
    """

    def __init__(
        self,
        p: int = 1,
        intercept: bool = True,
        n_components: int = 2,
        seed: int = 123,
        ridge_beta: float = 1e-6,
        ridge_sigma: float = 1e-6,
        dirichlet_alpha: float = 1.0,
    ):
        if n_components < 2:
            raise ValueError("n_components must be at least 2.")
        self.p = p
        self.intercept = intercept
        self.K = int(n_components)
        self.rng = np.random.default_rng(seed)
        self.base_var = BaseVAR(p=p, intercept=intercept)

        self.ridge_beta = ridge_beta
        self.ridge_sigma = ridge_sigma
        self.dirichlet_alpha = dirichlet_alpha

        self.result_: Optional[MixtureVARResult] = None

    def fit(
        self,
        data: pd.DataFrame,
        date_col: str = "date",
        n_iter: int = 1500,
        burn: int = 500,
    ) -> MixtureVARResult:
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
        beta_t, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (k_features, d)
        beta = beta_t.T                                       # (d, k_features)

        fitted_values = np.einsum("tk,kd->td", X, beta_t)
        residuals = Y - fitted_values

        # initialize latent regimes
        z = self.rng.integers(0, self.K, size=T_eff)
        pi = np.ones(self.K) / self.K

        # initialize regime covariances
        base_sigma = np.einsum("ti,tj->ij", residuals, residuals) / max(T_eff - k_features, 1)
        base_sigma = base_sigma + self.ridge_sigma * np.eye(d)
        Sigma = np.stack([base_sigma.copy() for _ in range(self.K)], axis=0)  # (K, d, d)

        beta_draws: list[np.ndarray] = []
        sigma_draws: list[np.ndarray] = []
        pi_draws: list[np.ndarray] = []
        z_draws: list[np.ndarray] = []

        fitted_values_last = None
        residuals_last = None

        for it in range(n_iter):
            # ---------------------------------------
            # 1) beta | z, Sigma, Y
            # Weighted LS with regime-specific scalar precision proxy
            # ---------------------------------------
            weights = np.zeros(T_eff, dtype=np.float64)
            for t in range(T_eff):
                k_t = z[t]
                # simple scalar precision proxy from regime covariance
                scale_t = np.mean(np.diag(Sigma[k_t]))
                weights[t] = 1.0 / max(scale_t, 1e-8)

            W_sqrt = np.sqrt(weights)[:, None]
            Xw = X * W_sqrt
            Yw = Y * W_sqrt

            XtX = np.einsum("ti,tj->ij", Xw, Xw) + self.ridge_beta * np.eye(k_features)
            XtY = np.einsum("ti,td->id", Xw, Yw)

            beta_t = np.linalg.solve(XtX, XtY)
            beta = beta_t.T

            fitted_values = np.einsum("tk,kd->td", X, beta_t)
            residuals = Y - fitted_values

            # ---------------------------------------
            # 2) z_t | beta, Sigma, pi
            # ---------------------------------------
            for t in range(T_eff):
                e_t = residuals[t]
                log_probs = np.zeros(self.K, dtype=np.float64)

                for k in range(self.K):
                    Sigma_k = Sigma[k]
                    Sigma_k_inv = np.linalg.inv(Sigma_k)

                    sign, logdet = np.linalg.slogdet(Sigma_k)
                    if sign <= 0:
                        raise RuntimeError(f"Non-PD Sigma[{k}] at iteration {it}.")

                    quad = float(np.einsum("i,ij,j->", e_t, Sigma_k_inv, e_t))
                    log_probs[k] = np.log(pi[k] + 1e-16) - 0.5 * (logdet + quad)

                # stabilize log probs
                log_probs -= log_probs.max()
                probs = np.exp(log_probs)
                probs /= probs.sum()

                z[t] = self.rng.choice(self.K, p=probs)

            # ---------------------------------------
            # 3) pi | z
            # ---------------------------------------
            counts = np.bincount(z, minlength=self.K)
            alpha_post = self.dirichlet_alpha + counts
            pi = self.rng.dirichlet(alpha_post)

            # ---------------------------------------
            # 4) Sigma_k | z, beta, Y
            # ---------------------------------------
            for k in range(self.K):
                idx = np.where(z == k)[0]

                if len(idx) == 0:
                    # avoid empty cluster collapse
                    Sigma[k] = base_sigma.copy()
                    continue

                E_k = residuals[idx]
                S_k = np.einsum("ti,tj->ij", E_k, E_k) / len(idx)
                Sigma[k] = S_k + self.ridge_sigma * np.eye(d)

            fitted_values_last = fitted_values
            residuals_last = residuals

            if it >= burn:
                beta_draws.append(beta.copy())
                sigma_draws.append(Sigma.copy())
                pi_draws.append(pi.copy())
                z_draws.append(z.copy())

        self.result_ = MixtureVARResult(
            beta_draws=beta_draws,
            sigma_draws=sigma_draws,
            pi_draws=pi_draws,
            z_draws=z_draws,
            fitted_values_last=fitted_values_last,
            residuals_last=residuals_last,
            design=design,
            n_components=self.K,
        )
        return self.result_

    def summary(self) -> dict:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return {
            "p": self.p,
            "intercept": self.intercept,
            "n_components": self.K,
            "n_saved_draws": len(self.result_.beta_draws),
            "n_obs_effective": self.result_.design.Y.shape[0],
            "n_series": self.result_.design.d,
            "n_features": self.result_.design.X.shape[1],
            "beta_draw_shape": self.result_.beta_draws[-1].shape,
            "sigma_draw_shape": self.result_.sigma_draws[-1].shape,
            "pi_draw_shape": self.result_.pi_draws[-1].shape,
            "z_draw_shape": self.result_.z_draws[-1].shape,
        }

    def posterior_mean_beta(self) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return np.mean(np.stack(self.result_.beta_draws, axis=0), axis=0)

    def posterior_mean_pi(self) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return np.mean(np.stack(self.result_.pi_draws, axis=0), axis=0)

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
            pi = self.result_.pi_draws[s]
            Sigma = self.result_.sigma_draws[s]

            mean = np.einsum("dk,k->d", beta, x)
            k = self.rng.choice(self.K, p=pi)

            sims[i] = self.rng.multivariate_normal(mean, Sigma[k])

        return sims