import numpy as np


def inv_wishart(df, scale):
    d = scale.shape[0]
    A = np.random.randn(d, d)
    W = A @ A.T
    return np.linalg.inv(W + scale)


def mvn(mean, cov):
    return np.random.multivariate_normal(mean, cov)