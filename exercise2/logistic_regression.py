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


def apply_hypothesis(term):
    """
    Using that function we apply sigmoid to theta*x
    :param term:
    :return:
    """
    calc_exp = np.exp(term * (-1))
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
    for i in range(num_iters):
        for i in range(n + 1):
            term = x_inc.dot(theta)
            sigmoided = apply_hypothesis(term) - labels
            theta[i] = theta[i] - l_rate * (np.dot(sigmoided, x_inc[:, i]))
    return theta


def read_test_data():
    """
    Using that function you are able to read test data
    :return:
    """
    data = np.loadtxt("exercise2/data.txt", delimiter=',')
    x1, x2, test_labels = data[:, 0], data[:, 1], data[:, 2]
    test_x = np.column_stack((x1, x2))
    return test_x, test_labels


def predict(test_x, test_labels, theta):
    """
    Using that function we are able to make our predictions.

    :param test_x: x test data
    :param test_labels: test labels
    :param theta: theta vector
    :return:
    """
    mtest, ntest = test_x.shape
    x_test_inc = np.column_stack((np.ones(mtest), test_x))
    predicted = []
    for i in range(mtest):
        prob = sigmoid(theta=theta, x=x_test_inc[i, :])
        if prob >= 0.5:
            predicted.append(1)
        else:
            predicted.append(0)
    return np.mean(np.array(predicted) == test_labels)


if __name__ == "__main__":
    print("Load train data")
    x, labels = read_data()
    print("Load test data")
    test_x, test_labels = read_test_data()
    print("Apply stochastic gradient descent algorithm")
    theta = stochastic_gradient_descent(0.0001, x, labels, 55000)
    print("Prediction accuraccy {}".format(predict(test_x, test_labels, theta)))
