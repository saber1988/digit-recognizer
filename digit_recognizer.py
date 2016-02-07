__author__ = 'shidaiting01'

import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import data_extractor as data_extractor
import data_storer as data_storer


def svc_classify(my_train_data, my_train_label, my_test_data, svc_c):
    # clf = svm.SVC(C=svc_c, kernel='poly')
    clf = svm.SVC(C=svc_c)
    scores = cross_validation.cross_val_score(clf, my_train_data, my_train_label, cv=5)
    print("svc(C=%.1f) accuracy: %0.3f (+/- %0.3f)" % (svc_c, scores.mean(), scores.std() * 2))
    clf.fit(my_train_data, my_train_label)
    my_test_label = clf.predict(my_test_data)
    file_name = "svc_%.1f.csv" % svc_c
    data_storer.save_data(my_test_label, file_name)


def knn_classify(my_train_data, my_train_label, my_test_data, neighbors):
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    scores = cross_validation.cross_val_score(clf, my_train_data, my_train_label, cv=5)
    print("knn(%d) accuracy: %0.3f (+/- %0.3f)" % (neighbors, scores.mean(), scores.std() * 2))
    clf.fit(my_train_data, my_train_label)
    my_test_label = clf.predict(my_test_data)
    file_name = "knn_%d.csv" % neighbors
    data_storer.save_data(my_test_label, file_name)


def gaussian_nb_classify(my_train_data, my_train_label, my_test_data):
    clf = GaussianNB()
    scores = cross_validation.cross_val_score(clf, my_train_data, my_train_label, cv=5)
    print("Gaussian native bayes accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    clf.fit(my_train_data, my_train_label)
    my_test_label = clf.predict(my_test_data)
    file_name = "gaussian_nb.csv"
    data_storer.save_data(my_test_label, file_name)


def multinomial_nb_classify(my_train_data, my_train_label, my_test_data):
    clf = MultinomialNB(alpha=0.1)
    scores = cross_validation.cross_val_score(clf, my_train_data, my_train_label, cv=5)
    print("multinomial native bayes accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    clf.fit(my_train_data, my_train_label)
    my_test_label = clf.predict(my_test_data)
    file_name = "multinomial_nb.csv"
    data_storer.save_data(my_test_label, file_name)


def random_forest_classify(my_train_data, my_train_label, my_test_data, estimators):
    clf = RandomForestClassifier(n_estimators=estimators)
    scores = cross_validation.cross_val_score(clf, my_train_data, my_train_label, cv=5)
    print("random forest(%d) accuracy: %0.3f (+/- %0.3f)" % (estimators, scores.mean(), scores.std() * 2))
    clf.fit(my_train_data, my_train_label)
    my_test_label = clf.predict(my_test_data)
    file_name = "random_forest_%d.csv" % estimators
    data_storer.save_data(my_test_label, file_name)


def gradient_boosting_classify(my_train_data, my_train_label, my_test_data, estimators):
    clf = GradientBoostingClassifier(n_estimators=estimators)
    scores = cross_validation.cross_val_score(clf, my_train_data, my_train_label, cv=5)
    print("gradient boosting(%d) accuracy: %0.3f (+/- %0.3f)" % (estimators, scores.mean(), scores.std() * 2))
    clf.fit(my_train_data, my_train_label)
    my_test_label = clf.predict(my_test_data)
    file_name = "gradient_boosting_%d.csv" % estimators
    data_storer.save_data(my_test_label, file_name)

if __name__ == '__main__':
    train_data, train_label = data_extractor.load_train_data("train1.csv")
    test_data = data_extractor.load_test_data("test1.csv")

    # svc_classify(train_data, np.ravel(train_label), test_data, svc_c=12.0)
    svc_classify(train_data, np.ravel(train_label), test_data, svc_c=10.0)
    # svc_classify(train_data, np.ravel(train_label), test_data, svc_c=8.0)
    # svc_classify(train_data, np.ravel(train_label), test_data, svc_c=5.0)
    # svc_classify(train_data, np.ravel(train_label), test_data, svc_c=2.0)
    # svc_classify(train_data, np.ravel(train_label), test_data, svc_c=1.0)
    # svc_classify(train_data, np.ravel(train_label), test_data, svc_c=0.5)

    # knn_classify(train_data, np.ravel(train_label), test_data, 4)
    knn_classify(train_data, np.ravel(train_label), test_data, 5)
    # knn_classify(train_data, np.ravel(train_label), test_data, 6)
    # knn_classify(train_data, np.ravel(train_label), test_data, 7)
    # knn_classify(train_data, np.ravel(train_label), test_data, 8)
    # knn_classify(train_data, np.ravel(train_label), test_data, 10)

    # gaussian_nb_classify(train_data, np.ravel(train_label), test_data)
    # multinomial_nb_classify(train_data, np.ravel(train_label), test_data)

    random_forest_classify(train_data, np.ravel(train_label), test_data, 100)
    gradient_boosting_classify(train_data, np.ravel(train_label), test_data, 100)
