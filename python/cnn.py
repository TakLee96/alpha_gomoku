""" Convolutional Neural Network for Policy and Value """
import numpy as np
import tensorflow as tf
from os import mkdir
from time import time
from scipy.io import loadmat
from sdknet import build_network

""" Hyperparameters """
LR = 1e-3
LAMBDA = 1e-3
EPSILON = 1e-7
MAX_STEPS = 30000
BATCH_SIZE = 500


class GomokuData():
    @staticmethod
    def _one_hot(y):
        n = len(y)
        y_h = np.zeros(shape=(n, 225), dtype=np.float64)
        for i in range(n):
            y_h[i, y[i]] = 1.0
        return y_h

    def __init__(self):
        mat = loadmat("data/formatted")
        self.X_t = mat["X_t"]
        self.y_t = self._one_hot(mat["y_t"][0])
        self.X_v = mat["X_v"]
        self.y_v = self._one_hot(mat["y_v"][0])
        self.n, _ = self.X_t.shape
        self.m, _ = self.X_v.shape
        self.cursor = self.n

    def reset(self):
        self.cursor = 0

    def next_batch(self, size):
        if self.cursor + size > self.n:
            self.cursor = 0
            ordering = np.random.permutation(self.n)
            self.X_t = self.X_t[ordering, :]
            self.y_t = self.y_t[ordering, :]
        X_b = self.X_t[self.cursor:(self.cursor + size), :]
        y_b = self.y_t[self.cursor:(self.cursor + size), :]
        self.cursor += size
        return X_b, y_b

    def random_test(self, size):
        which = np.random.choice(self.m, size, replace=False)
        X_b = self.X_v[which, :]
        y_b = self.y_v[which, :]
        return X_b, y_b

def train():
    with tf.Session() as sess:
        now = time()
        name = build_network(LAMBDA, EPSILON, LR)
        sess.run(tf.global_variables_initializer())
        data = GomokuData()
        s = tf.train.Saver()
        mkdir("model/%s")
        s.export_meta_graph("model/%s/%s.meta" % (name, name), clear_devices=True)

        train_step = lambda x, y: sess.run("train_step", feed_dict={"x:0": x, "y_:0": y, "is_training:0": True})
        accuracy = lambda x, y: sess.run("accuracy:0", feed_dict={"x:0": x, "y_:0": y, "is_training:0": False})
        save = lambda i: s.save(sess, "model/%s/%s" % (name, name), global_step=i, write_meta_graph=False)

        for i in xrange(MAX_STEPS):
            x_b, y_b = data.next_batch(BATCH_SIZE)
            if i % 100 == 0:
                print "step %d accuracy %g [%g sec]" % (i, accuracy(x_b, y_b), time() - now)
                now = time()
            train_step(x_b, y_b)
            if i % 1000 == 0:
                save(i)
                x_t, y_t = data.random_test(10 * BATCH_SIZE)
                print "===> validation accuracy %g (model saved)" % accuracy(x_t, y_t)
        
        save(MAX_STEPS)
        rates = np.zeros(shape=10, dtype=float)
        for j in xrange(10):
            x_t, y_t = data.random_test(10 * BATCH_SIZE)
            rates[j] = accuracy(x_t, y_t)
        print "===> validation accuracy %g (model saved)" % rates.mean()

def check():
    with tf.Session() as session:
        data = GomokuData()
        saver = tf.train.import_meta_graph("model/sdknet/sdknet.meta", clear_devices=True)
        # saver.restore(session, tf.train.latest_checkpoint("model/sdknet"))
        saver.restore(session, "model/sdknet/sdknet-25000")
        check_size = 10 * BATCH_SIZE
        num_checks = data.m / check_size
        accuracy = np.zeros(shape=num_checks, dtype=float)
        print "===> begin a total of %d patches" % num_checks
        for i in range(num_checks):
            x_t = data.X_t[(i * check_size):min((i+1)*check_size, data.m), :]
            y_t = data.y_t[(i * check_size):min((i+1)*check_size, data.m), :]
            accuracy[i] = session.run("accuracy:0", feed_dict={"x:0": x_t, "y_:0": y_t, "is_training:0": False})
            print "patch %d has accuracy %g" % (i, accuracy[i])
        print "===> overall test accuracy %g" % np.mean(accuracy)

check()

"""
21000: 0.464068
22000: 0.462516
23000: 0.465442
24000: 0.465711
25000: 
"""
