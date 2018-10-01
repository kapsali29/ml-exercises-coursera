from collections import Counter

import scipy.io
import numpy as np


def read_data():
    """
    Using that function
    :return:
    """
    mat_data = scipy.io.loadmat('exercise3/Data/ex3data1.mat')
    x_data = mat_data['X']
    y_labels = mat_data['y'][:, 0].ravel()
    return x_data, y_labels


def split_data(x_data, y_labels):
    """
    The following function split the data to train and test randomly
    :param x_data: train dataset
    :param y_labels: train labels
    :return:
    """
    indices = np.random.permutation(x_data.shape[0])
    training_idx, test_idx = indices[:4750], indices[4750:]
    x_train, x_test = x_data[training_idx, :], x_data[test_idx, :]
    y_train = y_labels[training_idx]
    y_test = y_labels[test_idx]
    return x_train, y_train, x_test, y_test


def apply_hypothesis(term):
    """
    Using that function we apply sigmoid to theta*x
    :param term:
    :return:
    """
    calc_exp = np.exp(term * (-1))
    sig = 1 / (1 + calc_exp)
    return sig


def sigmoid(theta, x):
    """
    The following function calculates the sigmoid function
    :param theta: theta vector parameters
    :param x: input values
    :return:
    """
    calc_exp = np.exp(np.dot(theta, x) * (-1))
    sig = 1 / (1 + calc_exp)
    return sig


def mean_normalize(X):
    '''apply mean normalization to each column of the matrix X'''
    X[X == 0.] = 1e-6
    X_max = X.max(axis=0)
    return X - X_max


def one_vs_all_gradient_descent(train_data, train_labels, num_iters, a, l2):
    """
    Using that function the one_vs_all gradient descent algorithm is computed
    :param train_data: train dataset
    :param train_labels: train labels
    :param num_iters: number of max iterations
    :param a: learning rate a
    :param l2: l2 penalty
    :return: theta parameters
    """
    m, n = train_data.shape
    unique_classes = [int(n) for n in np.unique(train_labels)]
    theta = np.zeros((len(unique_classes), n + 1))
    train_data_inc = np.column_stack((np.ones(m), train_data))
    train_data_inc = mean_normalize(train_data_inc)
    for i in unique_classes:
        print("Train classifier for {} class".format(i))
        labels_cp = (train_labels == i).astype(int)

        theta_current = theta[i - 1, :]
        for n in range(num_iters):
            term = np.dot(train_data_inc, theta_current)
            sigmoided = apply_hypothesis(term) - labels_cp
            new_theta = (-1) * (a / m) * np.dot(sigmoided, train_data_inc)
            theta_current[0] = theta_current[0] + new_theta[0]
            theta_current[1:] = theta_current[1:] + new_theta[1:] - (l2 / m) * theta_current[1:]
        theta[i - 1, :] = theta_current
    return theta


def cost(theta, X, y):
    """
    The following function computes the cost for each classifier
    :param theta: current theta
    :param X: x input
    :param y: y label
    :return:
    """
    predictions = sigmoid(X, theta)
    # predictions[predictions == 1.0] = 0.999  # log(1)=0 causes error in division
    error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
    return sum(error) / len(y)


def one_vs_all_predict(classifiers, x_test, y_test):
    """

    :param classifiers: theta parameters for each category
    :param x_test: test data
    :param y_test: test labels
    :return:
    """
    classified = []
    mtest, ntest = x_test.shape
    x_test_inc = np.column_stack((np.ones(mtest), x_test))
    x_test_inc = mean_normalize(x_test_inc)
    for i in range(x_test_inc.shape[0]):
        x_current = x_test_inc[i, :]
        max = 0
        pos = 0
        for j in range(classifiers.shape[0]):
            y = (y_test == j + 1).astype(int)
            theta = classifiers[j, :]
            ans = cost(theta, x_current, y)
            if ans >= max:
                max = ans
                pos = j + 1
        classified.append(pos)
    return (np.mean(np.array(classified) == y_test)) * 100


if __name__ == "__main__":
    x_data, y_labels = read_data()
    x_train, y_train, x_test, y_test = split_data(x_data, y_labels)
    classifiers = one_vs_all_gradient_descent(x_train, y_train, 1000, 0.1, 0.1)
    print(one_vs_all_predict(classifiers, x_test, y_test))
