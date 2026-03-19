import numpy as np
from models.base_bvar import BaseVAR
from utils.priors import minnesota_prior
from utils.sampling import inv_wishart


class GaussianBVAR(BaseVAR):

    def fit(self, Y, n_iter=1000):
        X, Yt = self.build_design(Y)
        T, d = Yt.shape
        k = X.shape[1]

        beta = np.zeros((d, k))
        Sigma = np.eye(d)

        beta_draws = []
        sigma_draws = []

        prior_mean, prior_var = minnesota_prior(k, d)

        for _ in range(n_iter):

            # Sample beta (OLS-style)
            XtX = X.T @ X
            XtY = X.T @ Yt

            beta = np.linalg.solve(XtX + np.eye(k), XtY).T

            # Sample Sigma
            residuals = Yt - X @ beta.T
            Sigma = inv_wishart(d + T, residuals.T @ residuals)

            beta_draws.append(beta.copy())
            sigma_draws.append(Sigma.copy())

        self.beta_draws = beta_draws
        self.sigma_draws = sigma_draws