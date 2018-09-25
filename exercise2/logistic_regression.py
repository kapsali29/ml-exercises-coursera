import numpy as np


def read_data():
    """
    The following function load the data from file
    :return:
    """
    data = np.loadtxt("exercise2/data.txt", delimiter=',')
    x1, x2, labels = data[:, 0], data[:, 1], data[:, 2]
    x = np.column_stack((x1, x2))
    return x, labels


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


def stochastic_gradient_descent(l_rate, x, labels, num_iters):
    """
    The following function implements the stochastic gradient descent algorithm
    to find theta parameters

    :param l_rate: learning rate
    :param x: input vector
    :param labels: output labels
    :return:
    """
    m, n = x.shape
    theta = np.zeros(n + 1)
    x_inc = np.column_stack((np.ones(m), x))
