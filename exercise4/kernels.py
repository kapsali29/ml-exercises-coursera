import numpy as np


def gaussianKernel(x1, x2, sigma):
    """
    Computes the radial basis function
    Returns a radial basis function kernel between x1 and x2.

    Parameters
    ----------
    x1 :  numpy ndarray
        A vector of size (n, ), representing the first datapoint.

    x2 : numpy ndarray
        A vector of size (n, ), representing the second datapoint.

    sigma : float
        The bandwidth parameter for the Gaussian kernel.

    Returns
    -------
    sim : float
        The computed RBF between the two provided data points.
    """
    sim = np.exp((-1) * ((np.linalg.norm(x1 - x2, ord=2) ** 2) / (2) * (sigma ** 2)))
    return sim


if __name__ == "__main__":
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = gaussianKernel(x1, x2, sigma)
    print(sim)
