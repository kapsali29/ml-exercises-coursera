import os
import numpy as np
from scipy.io import loadmat


def _data():
    """
    This function is used to load the ex7data2.mat dataset
    :return:
    """
    work_dir = os.getcwd()
    data_folder = os.path.join(work_dir, "Data/ex7data1.mat")
    data = loadmat(data_folder)
    return data['X']


def compute_covariance_matrix():
    """
    The following function is used to compute the covariance matrix using the following formula
    sigma = (1/m)X.T*X, X: (m,n)
    :return: covariance matrix sigma: sigma (n,n)
    """
    X = _data()
    m, n = X.shape
    sigma = (1 / m) * np.dot(X.T, X)
    return sigma


def compute_project_matrix():
    """
    The following function computes X's projection matrix using svd and finds the best parameter K for pca
    :return: projection matrix and K
    """
    sigma = compute_covariance_matrix()
    U, S, V = np.linalg.svd(sigma)
    print(V)


compute_project_matrix()
