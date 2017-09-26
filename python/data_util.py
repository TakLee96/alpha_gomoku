""" utilities for data processing """

import numpy as np

def one_hot(action):
    h = np.zeros(shape=225, dtype=np.float32)
    h[np.ravel_multi_index(action, dims=(15, 15))] = 1.0
    return h

def one_hot_batch(actions):
    n = actions.shape[0]
    h = np.zeros(shape=(n, 225), dtype=np.float32)
    for i in range(n):
        h[n, np.ravel_multi_index(actions[n], dims=(15, 15))] = 1.0
    return h

def dist_to_prob(dist):
    h = np.zeros(shape=225, dtype=np.float32)
    for x, y, p in dist:
        h[np.ravel_multi_index((x, y), dims=(15, 15))] = p
    return h

class Data():
    def __init__(self, X, Y):
        assert X.shape[0] == Y.shape[0]
        n = X.shape[0]
        ordering = np.random.permutation(n)
        X = X[ordering, :]
        Y = Y[ordering, :]
        self.m = n // 10
        self.n = n - self.m
        self.X_t = X[self.m:, :]
        self.Y_t = Y[self.m:, :]
        self.X_v = X[:self.m, :]
        self.Y_v = Y[:self.m, :]
        self.cursor = 0

    def next_batch(self, size):
        if self.cursor + size > self.n:
            self.cursor = 0
            ordering = np.random.permutation(self.n)
            self.X_t = self.X_t[ordering, :]
            self.Y_t = self.Y_t[ordering, :]
        X_b = self.X_t[self.cursor:(self.cursor + size), :]
        Y_b = self.Y_t[self.cursor:(self.cursor + size), :]
        self.cursor += size
        return X_b, Y_b

    def test_batch(self, size):
        which = np.random.choice(self.m, size, replace=False)
        X_b = self.X_v[which, :]
        Y_b = self.Y_v[which, :]
        return X_b, Y_b
