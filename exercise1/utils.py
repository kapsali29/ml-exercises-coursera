import numpy as np


def warm_up():
    """
    Using that function we create an eye array
    :return:
    """
    return np.eye(5)


def cost_function(m, x, y, theta0, theta1):
    """
    Using that function we calculate cost function
    :param m: number of observations
    :param x: input x
    :param y: output y
    :param theta0: parameter
    :param theta1: parameter
    :return: J(theta0, theta1)
    """
    temp_a = np.sum(np.square((theta0 + theta1 * x) - y))
    J = (1 / 2 * m) * temp_a
    return J


x = np.array([1, 1, 1])
y = np.array([1, 1, 1])

print(cost_function(3, x, y, 1, 1))
