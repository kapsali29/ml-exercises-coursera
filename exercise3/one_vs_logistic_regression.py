import scipy.io
import numpy as np


def read_data():
    """
    Using that function
    :return:
    """
    mat_data = scipy.io.loadmat('Data/ex3data1.mat')
    train_data = mat_data['X']
    train_labels = mat_data['y']
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


def one_vs_all_gradient_descent(train_data, train_labels):
    m, n = train_data.shape
    labels_num = train_labels.shape[0]
    theta = np.zeros((labels_num, n + 1))
    train_data_inc = np.column_stack((np.ones(m), train_data))
    labels_cp = train_labels
    print(labels_cp)
    for i in range(labels_num):
        labels_cp = train_labels
        labels_cp[labels_cp == i] = 1
        labels_cp[labels_cp != i] = 0
        theta_current = theta[i, :]
        term = np.dot(train_data_inc, theta_current)
        sigmoided = apply_hypothesis(term) - labels_cp
        print(sigmoided)
        break


train_data, train_labels = read_data()
one_vs_all_gradient_descent(train_data, train_labels)
