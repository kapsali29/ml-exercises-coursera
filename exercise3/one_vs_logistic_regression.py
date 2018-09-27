import scipy.io
import numpy as np


def read_data():
    """
    Using that function
    :return:
    """
    mat_data = scipy.io.loadmat('Data/ex3data1.mat')
    train_data = mat_data['X']
    train_labels = mat_data['y'][:, 0]
    return train_data, train_labels


def apply_hypothesis(term):
    """
    Using that function we apply sigmoid to theta*x
    :param term:
    :return:
    """
    calc_exp = np.exp(term * (-1))
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
    unique_classes = np.unique(train_labels)
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


train_data, train_labels = read_data()
print(one_vs_all_gradient_descent(train_data, train_labels, 100, 0.0001, 1).shape)
