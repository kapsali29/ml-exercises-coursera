import os
import numpy as np
from scipy.io import loadmat


class PCA(object):

    def normalize_data(self, X):
        """
        The following function is used to normalize the data using the following formula:
        xi:= (xi-mi)/si
        :param X: X input data
        :return: normalized dataset
        """
        means = np.mean(X, axis=1)
        stds = np.std(X, axis=1)
        normalized_data = np.zeros(X.shape)
        for i in range(len(stds)):
            normalized_data[i, :] = (X[i, :] - means[i]) / stds[i] + 1e-7
        return normalized_data

    def _data(self):
        """
        This function is used to load the ex7data2.mat dataset
        :return:
        """
        work_dir = os.getcwd()
        data_folder = os.path.join(work_dir, "Data/ex7data1.mat")
        data = loadmat(data_folder)['X']
        increased_data = np.column_stack((data, data[:, 1]))
        normalized_data = self.normalize_data(increased_data)
        return normalized_data

    def compute_covariance_matrix(self):
        """
        The following function is used to compute the covariance matrix using the following formula
        sigma = (1/m)X.T*X, X: (m,n)
        :return: covariance matrix sigma: sigma (n,n)
        """
        X = self._data()
        m, n = X.shape
        sigma = (1 / m) * np.dot(X.T, X)
        return sigma

    def compute_project_matrix(self):
        """
        The following function computes X's projection matrix using svd and finds the best parameter K for pca
        :return: projection matrix and K
        """
        sigma = self.compute_covariance_matrix()
        U, S, V = np.linalg.svd(sigma)
        max_k = U.shape[1]
        diagonal_U = np.diag(U)
        diagonal_sum = np.sum(diagonal_U)
        u_sum = np.zeros(max_k)
        for i in range(max_k):
            u_sum[i] = np.sum(diagonal_U[:i + 1]) / diagonal_sum
        index = np.argmax(u_sum)
        best_k = index + 1
        u_reduced = U[:, :best_k]
        return u_reduced.T

    def compute_z_matrix(self):
        """
        The following function returns z data
        :return:
        """
        X = self._data()
        u_reduced_trans = self.compute_project_matrix()
        z_matrix = np.dot(u_reduced_trans, X.T).T
        return z_matrix


if __name__ == "__main__":
    pca = PCA()
    print(pca.compute_z_matrix())
