import numpy as np


def minnesota_prior(k, d, lambda_own=0.2):
    mean = np.zeros((d, k))
    var = np.ones((d, k)) * lambda_own**2

    return mean, var