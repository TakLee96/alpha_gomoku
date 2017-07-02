import numpy as np
from os import path
from sys import argv
from sklearn.svm import SVR
from scipy.io import loadmat
from sklearn.externals import joblib


def train(dataset, kernel, degree):
    m = loadmat(path.join(path.dirname(__file__), "data", dataset))
    X_t = m["X_t"].astype(np.float64)
    y_t = m["y_t"][0].astype(np.float64)
    X_v = m["X_v"].astype(np.float64)
    y_v = m["y_v"][0].astype(np.float64)
    del m

    print("training begin")
    clf = SVR(kernel=kernel, degree=degree)
    clf.fit(X_t, y_t)
    print("training complete")

    joblib.dump(clf, path.join(path.dirname(__file__), "model", "value", "svm",
        "%s-%s-%d.pkl" % (dataset, kernel, degree)))
    print("model saved to disk")

    score_v = clf.score(X_v, y_v)
    print("validation score: %g" % score_v)
    score_t = clf.score(X_t, y_t)
    print("training score: %g" % score_t)


def check(dataset, kernel, degree):
    m = loadmat(path.join(path.dirname(__file__), "data", dataset))
    X_t = m["X_t"].astype(np.float64)
    y_t = m["y_t"][0].astype(np.float64)
    X_v = m["X_v"].astype(np.float64)
    y_v = m["y_v"][0].astype(np.float64)
    del m
    
    clf = joblib.load(path.join(path.dirname(__file__), "model", "value", "svm",
        "%s-%s-%d.pkl" % (dataset, kernel, degree)))
    score_v = clf.score(X_v, y_v)
    print("validation score: %g" % score_v)
    score_t = clf.score(X_t, y_t)
    print("training score: %g" % score_t)
    

if __name__ == "__main__":
    if len(argv) != 5 or argv[1] not in ("train", "check"):
        print("Usage: python value_svm.py [train/check] [dataset] [kernel] [degree]")
    else:
        command = argv[1]
        dataset = argv[2]
        kernel = argv[3]
        degree = int(argv[4])
        if command == "train":
            train(dataset, kernel, degree)
        else:
            check(dataset, kernel, degree)
