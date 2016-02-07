__author__ = 'shidaiting01'
import csv
import numpy as np

def load_test_data(test_file):
    data = []
    with open(test_file, 'rb') as my_file:
        lines = csv.reader(my_file)
        for line in lines:
            data.append(line)

    data.remove(data[0])
    return normalize(to_int(np.mat(data)))


def load_train_data(train_file):
    data = []
    with open(train_file, 'rb') as my_file:
        lines = csv.reader(my_file)
        for line in lines:
            data.append(line)

    data.remove(data[0])
    data = np.array(data)
    label = data[:, 0]
    data = data[:, 1:]
    return normalize(to_int(np.mat(data))), to_int(np.mat(label))


def to_int(data_mat):
    m, n = np.shape(data_mat)
    new_mat = np.zeros((m, n))
    for i in xrange(m):
        for j in xrange(n):
            new_mat[i, j] = int(data_mat[i, j])
    return new_mat


def normalize(data_mat):
    m, n = np.shape(data_mat)
    new_mat = np.zeros((m, n))
    for i in xrange(m):
        for j in xrange(n):
            if data_mat[i, j] != 0:
                new_mat[i, j] = 1
    return new_mat

if __name__ == '__main__':
    train_data, train_label = load_train_data("train1.csv")
    print np.shape(train_label)
    print np.shape(train_data)