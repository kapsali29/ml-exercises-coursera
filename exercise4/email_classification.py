import random

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from os.path import isfile, join


class SpamHamFiltering(object):

    def split_data(self):
        """
        Using that function you can split test and train data files
        :return:
        """
        work_dir = os.getcwd()
        ham_path = join(work_dir, 'Data/enron1/ham')
        spam_path = join(work_dir, 'Data/enron1/spam')
        ham_files_ini = [f for f in os.listdir(ham_path) if isfile(join(ham_path, f))]
        ham_files = random.sample(ham_files_ini, 1500)
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

    def process_emails(self, emails):
        """
        Using that function you can process and clean emails contents
        :return:
        """
        ps = PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        processed_data = []
        for email in emails:
            new_mail = re.sub(r'http\S+', 'httpaddr', email)
            new_mail = re.sub(r'$', 'dollar', new_mail)
            new_mail = re.sub(r'%', 'percentage', new_mail)
            new_mail = re.sub('<[^<]+?>', '', new_mail)
            new_mail = re.sub('\d+', ' number', new_mail)
            new_mail = re.sub(r'[\w\.-]+@[\w\.-]+', 'emailaddr', new_mail)
            words = tokenizer.tokenize(new_mail)
            processed_data.append(
                [ps.stem(re.compile('[^a-zA-Z0-9]').sub('', word.lower()).strip()) for word in words if
                 word not in stop_words and len(word) > 2])
        return processed_data

    def train_svm_classifier(self, x_train, y_train):
        """
        That function trains
        :param x_train:
        :param y_train:
        :return:
        """
        clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', LinearSVC()),
                        ])
        data = [" ".join(text) for text in x_train]
        clf.fit(data, y_train)
        return clf

    def predict(self, x_test, y_test, clf):
        """
        Using the following function we can predict if test data belong to ham or spam class
        :param x_test: x test dataset
        :param y_test: labels
        :param clf: classifier SVM with Gaussian kernels
        :return: accuracy
        """
        test_data = self.process_emails(x_test)
        data = [" ".join(text) for text in test_data]
        predicted = clf.predict(data)
        return np.mean(predicted == y_test)

    def operate(self):
        """
        That function performs the email classification steps
        :return:
        """
        x_train, y_train, x_test, y_test = self.split_data()
        proc_data = self.process_emails(x_train)
        clf = self.train_svm_classifier(proc_data, y_train)
        print(self.predict(x_test, y_test, clf))


if __name__ == "__main__":
    filtering = SpamHamFiltering()
    filtering.operate()
