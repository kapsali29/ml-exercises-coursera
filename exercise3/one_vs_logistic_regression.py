import scipy.io
import numpy as np


def read_data():
    """
    Using that function
    :return:
    """
    mat_data = scipy.io.loadmat('Data/ex3data1.mat')
    x_data = mat_data['X']
    y_labels = mat_data['y'][:, 0]
    return x_data, y_labels


def split_data(x_data, y_labels):
    """
    The following function split the data to train and test randomly
    :param x_data: train dataset
    :param y_labels: train labels
    :return:
    """
    total_data = np.column_stack((x_data, y_labels))
    shuffled_data = np.random.permutation(total_data)
    train = shuffled_data[0:4500, :]
    test = shuffled_data[4500:, :]
    x_train = train[:, 0:x_data.shape[1]]
    y_train = train[:, x_data.shape[1]].astype(int)
    x_test = test[:, 0:x_data.shape[1]]
    y_test = test[:, x_data.shape[1]].astype(int)
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
    print(unique_classes)
    theta = np.zeros((len(unique_classes), n + 1))
    train_data_inc = np.column_stack((np.ones(m), train_data))
    for i in unique_classes:
        print("Train classifier for {} class".format(i))
        labels_cp = train_labels
        labels_cp[labels_cp == i] = 1
        labels_cp[labels_cp != i] = 0

        theta_current = theta[i - 1, :]
        for n in range(num_iters):
            term = np.dot(train_data_inc, theta_current)
            sigmoided = apply_hypothesis(term) - labels_cp
            new_theta = (-a / m) * np.dot(sigmoided, train_data_inc)

            theta_current[0] = theta_current[0] + new_theta[0]
            theta_current[1:] = theta_current[1:] + new_theta[1:] - (l2 / m) * theta_current[1:]
        theta[i - 1, :] = theta_current
    return theta


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
    for i in range(x_test_inc.shape[0]):
        x_current = x_test_inc[i, :]

        for j in range(classifiers.shape[0]):
            theta = classifiers[j, :]
            goes = sigmoid(theta, x_current)
            tmp_list.append(goes)
        classified.append(max(tmp_list))
    return np.mean(np.array(classified) == y_test)


x_data, y_labels = read_data()
x_train, y_train, x_test, y_test = split_data(x_data, y_labels)
classifiers = one_vs_all_gradient_descent(x_train, y_train, 1000, 0.001, 1)
print(one_vs_all_predict(classifiers, x_test, y_test))
