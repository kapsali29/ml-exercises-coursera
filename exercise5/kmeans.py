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


def computeCentroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the data points
    assigned to each centroid.

    Parameters
    ----------
    X : array_like
        The datset where each row is a single data point. That is, it
        is a matrix of size (m, n) where there are m datapoints each
        having n dimensions.

    idx : array_like
        A vector (size m) of centroid assignments (i.e. each entry in range [0 ... K-1])
        for each example.

    K : int
        Number of clusters

    Returns
    -------
    centroids : array_like
        A matrix of size (K, n) where each row is the mean of the data
        points assigned to it.

    """
    # Useful variables
    m, n = X.shape
    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))
    for i in range(K):
        ix_current = np.isin(idx, i)
        indices = np.where(ix_current)
        centroids[i] = sum(X[indices]) / len(X[indices])
    return centroids


def kMeansInitCentroids(X, K):
    """
    This function initializes K centroids that are to be used in K-means on the dataset x.

    Parameters
    ----------
    X : array_like
        The dataset of size (m x n).

    K : int
        The number of clusters.

    Returns
    -------
    centroids : array_like
        Centroids of the clusters. This is a matrix of size (K x n).

    Instructions
    ------------
    You should set centroids to randomly chosen examples from the dataset X.
    """
    m, n = X.shape

    # You should return this values correctly
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K], :]
    return centroids


def runkMeans(X, centroids, max_iters=10):
    """
    Runs the K-means algorithm.
    """
    K = centroids.shape[0]
    idx = None
    idx_history = []
    centroid_history = []

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        idx_history.append(idx)
        centroid_history.append(centroids)
        centroids = computeCentroids(X, idx, K)
    return centroids, idx


if __name__ == "__main__":
    print("Load ex7data2.mat dataset")
    X = load_data()
    K = 3
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    print("Compute the centroid memberships for every example")
    idx = findClosestCentroids(X, initial_centroids)
    print("Compute new centroids")
    centroids = computeCentroids(X, idx, K)
    print(centroids)
