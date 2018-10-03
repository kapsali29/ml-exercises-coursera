import numpy as np
import os
import scipy.io
from scipy.io import loadmat


def load_data():
    """
    This function is used to load the ex7data2.mat dataset
    :return:
    """
    work_dir = os.getcwd()
    data_folder = os.path.join(work_dir, "Data/ex7data2.mat")
    data = loadmat(data_folder)
    return data['X']


def findClosestCentroids(X, centroids):
    """
    Computes the centroid memberships for every example.

    Parameters
    ----------
    X : array_like
        The dataset of size (m, n) where each row is a single example.
        That is, we have m examples each of n dimensions.

    centroids : array_like
        The k-means centroids of size (K, n). K is the number
        of clusters, and n is the the data dimension.

    Returns
    -------
    idx : array_like
        A vector of size (m, ) which holds the centroids assignment for each
        example (row) in the dataset X.

    Instructions
    ------------
    Go over every example, find its closest centroid, and store
    the index inside `idx` at the appropriate location.
    Concretely, idx[i] should contain the index of the centroid
    closest to example i. Hence, it should be a value in the
    range 0..K-1

    Note
    ----
    You can use a for-loop over the examples to compute this.
    """
    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly.
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        current_x = X[i, :]
        norms_x = np.linalg.norm(current_x - centroids, ord=2, axis=1) ** 2
        idx[i] = np.argmin(norms_x)
    return idx


if __name__ == "__main__":
    print("Load ex7data2.mat dataset")
    X = load_data()
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    idx = findClosestCentroids(X, initial_centroids)
    print(idx[:3])
