# def email_pre_processing():
#
import os
from os.path import isfile, join


def load_data():
    """
    Using that function you can load test and train data
    :return:
    """
    work_dir = os.getcwd()
    ham_path = join(work_dir, 'Data/enron1/ham')
    spam_path = join(work_dir, 'Data/enron1/spam')
    ham_files = [f for f in os.listdir(ham_path) if isfile(join(ham_path, f))][0:1500]
    spam_files = [f for f in os.listdir(spam_path) if isfile(join(spam_path, f))]
    ham_train = ham_files[0:1350]
    ham_train_labels = ["ham"] * len(ham_train)
    ham_test = ham_files[1350:]
    ham_test_labels = ["ham"] * len(ham_test)
    spam_train = spam_files[0:1350]
    spam_train_labels = ["spam"] * len(spam_train)
    spam_test = spam_files[1350:0]
    spam_test_labels = ["spam"] * len(spam_test)
    x_train = ham_train + spam_train
    y_train = ham_train_labels + spam_train_labels
    x_test = ham_test + spam_test
    y_test = ham_test_labels + spam_test_labels
    return x_train, y_train, x_test, y_test


load_data()
