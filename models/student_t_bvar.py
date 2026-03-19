import numpy as np
from models.base_bvar import BaseVAR
from utils.sampling import inv_wishart


class StudentTBVAR(BaseVAR):

    def __init__(self, p=2, nu=8):
        super().__init__(p)
        self.nu = nu

    def fit(self, Y, n_iter=1500):
        X, Yt = self.build_design(Y)
        T, d = Yt.shape
        k = X.shape[1]

        beta = np.zeros((d, k))
        Sigma = np.eye(d)
        lam = np.ones(T)

        beta_draws = []

        for _ in range(n_iter):

            # Sample beta (weighted regression)
            W = np.diag(lam)
            XtW = X.T @ W

            beta = np.linalg.solve(XtW @ X + np.eye(k), XtW @ Yt).T

            # Residuals
            E = Yt - X @ beta.T

            # Sample lambda_t
            for t in range(T):
                quad = E[t] @ np.linalg.inv(Sigma) @ E[t]
                lam[t] = np.random.gamma((self.nu + d) / 2,
                                         2 / (self.nu + quad))

            # Sample Sigma
            Ew = E * np.sqrt(lam[:, None])
            Sigma = inv_wishart(d + T, Ew.T @ Ew)

            beta_draws.append(beta.copy())

        self.beta_draws = beta_draws