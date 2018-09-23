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


def gradient_descent(num_iters, m, x, y, l_rate):
    """
    That function computes the gradient descent algorithm
    :param num_iters: number of max iterations
    :param m: number of observations
    :param x: input
    :param y: expected output
    :param l_rate: learning rate
    :return:
    """
    theta0 = 0
    theta1 = 0
    for i in range(0, num_iters):
        theta0 = theta0 - (l_rate / m) * (np.sum((theta0 + theta1 * x) - y))
        theta1 = theta1 - (l_rate / m) * (np.sum(((theta0 + theta1 * x) - y) * x))
    return theta0, theta1


def hypothesis(theta0, theta1, x):
    """
    The following function calculates the linear regression hypothesis
    :param theta0: parameter theta0
    :param theta1: parameter theta1
    :param x: input x
    :return: return expected value close to real one
    """
    h = theta0 + theta1 * x
    return h


def read_data():
    """
    Using that function data are read and returned x, y vectors
    :return:
    """
    data = np.loadtxt("exercise1/data_linear.txt", delimiter=',')
    x, y = data[:, 0], data[:, 1]
    m = y.size
    return x, y, m


if __name__ == "__main__":
    print("Run warm up function")
    eye_matrix = warm_up()
    print("Read vectors x,y and get number of observations")
    x, y, m = read_data()
    print(x.size)
    print("Run gradient descent algorithm")
    num_iters = 2500
    l_rate = 0.01
    theta0, theta1 = gradient_descent(num_iters, m, x, y, l_rate)
    print("Show parameters from gradient descent")
    print("Theta0 parameter is {}".format(theta0))
    print("Theta1 parameter is {}".format(theta1))
    print("The hypothesis function will get the following format: H(x)={}+{}x".format(theta0, theta1))
