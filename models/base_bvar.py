import numpy as np


class BaseVAR:
    def __init__(self, p=2, intercept=True):
        self.p = p
        self.intercept = intercept

    def build_design(self, Y):
        T, d = Y.shape
        X = []

        for t in range(self.p, T):
            row = []
            if self.intercept:
                row.append(1.0)
            for lag in range(1, self.p + 1):
                row.extend(Y[t - lag])
            X.append(row)

        X = np.array(X)
        Y_target = Y[self.p:]

        return X, Y_target

    def predict_mean(self, beta, x):
        return beta @ x