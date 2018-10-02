# def email_pre_processing():
#
import os
from os.path import isfile, join


def split_data():
    """
    Using that function you can split test and train data files
    :return:
    """
    work_dir = os.getcwd()
    ham_path = join(work_dir, 'Data/enron1/ham')
    spam_path = join(work_dir, 'Data/enron1/spam')
    ham_files = [f for f in os.listdir(ham_path) if isfile(join(ham_path, f))][0:1500]
    spam_files = [f for f in os.listdir(spam_path) if isfile(join(spam_path, f))]
    ham_train = ham_files[0:1350]
    ham_train_data = []
    for ham_file in ham_train:
        with open(join(ham_path, ham_file), encoding="utf8", errors="ignore") as hf:
            ham_train_data.append(hf.read())
    ham_train_labels = ["ham"] * len(ham_train)
    ham_test = ham_files[1350:]
    ham_test_data = []
    for ham_file in ham_test:
        with open(join(ham_path, ham_file), encoding="utf8", errors="ignore") as hf:
            ham_test_data.append(hf.read())
    ham_test_labels = ["ham"] * len(ham_test)
    spam_train = spam_files[0:1350]
    spam_train_data = []
    for spam_file in spam_train:
        with open(join(spam_path, spam_file), encoding="utf8", errors="ignore") as sf:
            spam_train_data.append(sf.read())
    spam_train_labels = ["spam"] * len(spam_train)
    spam_test = spam_files[1350:0]
    spam_test_data = []
    for spam_file in spam_test:
        with open(join(spam_path, spam_file), encoding="utf8", errors="ignore") as sf:
            spam_test_data.append(sf.read())
    spam_test_labels = ["spam"] * len(spam_test)
    x_train = ham_train_data + spam_train_data
    y_train = ham_train_labels + spam_train_labels
    x_test = ham_test_data + spam_test_data
    y_test = ham_test_labels + spam_test_labels
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = split_data()

