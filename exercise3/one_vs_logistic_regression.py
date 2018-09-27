import scipy.io


def read_data():
    """
    Using that function
    :return:
    """
    mat_data = scipy.io.loadmat('exercise3/Data/ex3data1.mat')
    train_data = mat_data['X']
    train_labels = mat_data['y']
    return train_data, train_labels
