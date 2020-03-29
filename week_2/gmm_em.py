import numpy as np


def multivariate_normal_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    x = x.copy()

    if len(x.shape) == 1:
        x = x.reshape(1, -1)
        is_single = True
    else:
        is_single = False

    n = x.shape[1]
    multiplier = 1. / np.sqrt(np.linalg.det(cov) * (2 * np.pi) ** n)

    inv_cov = np.linalg.pinv(cov)
    delta = (x - mu)
    inv_cov_delta = np.einsum('kj,ij->ik', inv_cov, delta)
    exponent_arg = - (1 / 2) * np.einsum('ij,ij->i', delta, inv_cov_delta)
    result = multiplier * np.exp(exponent_arg)
    return result[0] if is_single else result
