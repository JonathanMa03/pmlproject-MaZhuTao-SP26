import numpy as np
from models.base_bvar import BaseVAR


class SVBVAR(BaseVAR):

    def __init__(self, p=2):
        super().__init__(p)

    def fit(self, Y, n_iter=1500):
        X, Yt = self.build_design(Y)
        T, d = Yt.shape
        k = X.shape[1]

        beta = np.zeros((d, k))
        h = np.zeros((T, d))  # log vol

        for _ in range(n_iter):

            # beta
            weights = np.exp(-h)
            W = np.diag(weights.mean(axis=1))

            beta = np.linalg.solve(X.T @ W @ X + np.eye(k), X.T @ W @ Yt).T

            E = Yt - X @ beta.T

            # update volatility (simple RW)
            h += 0.1 * np.random.randn(T, d)

        self.beta = beta