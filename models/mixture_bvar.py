import numpy as np
from models.base_bvar import BaseVAR
from utils.sampling import inv_wishart


class MixtureBVAR(BaseVAR):

    def __init__(self, p=2, K=2):
        super().__init__(p)
        self.K = K

    def fit(self, Y, n_iter=1500):
        X, Yt = self.build_design(Y)
        T, d = Yt.shape
        k = X.shape[1]

        beta = np.zeros((d, k))
        z = np.random.randint(0, self.K, size=T)
        pi = np.ones(self.K) / self.K
        Sigma = [np.eye(d) for _ in range(self.K)]

        for _ in range(n_iter):

            # beta
            beta = np.linalg.solve(X.T @ X + np.eye(k), X.T @ Yt).T

            E = Yt - X @ beta.T

            # update z
            for t in range(T):
                probs = []
                for k_ in range(self.K):
                    val = np.exp(-0.5 * E[t] @ np.linalg.inv(Sigma[k_]) @ E[t])
                    probs.append(pi[k_] * val)
                probs = np.array(probs) / np.sum(probs)
                z[t] = np.random.choice(self.K, p=probs)

            # update pi
            counts = np.bincount(z, minlength=self.K)
            pi = np.random.dirichlet(1 + counts)

            # update Sigma
            for k_ in range(self.K):
                Ek = E[z == k_]
                if len(Ek) > 0:
                    Sigma[k_] = inv_wishart(d + len(Ek), Ek.T @ Ek)

        self.beta = beta