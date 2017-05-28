""" neural network training """
import numpy as np
import tensorflow as tf
from os import path
from sys import argv
from time import time
from scipy.io import loadmat
from network.policy.deepsdknet import build_network


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
        mat = loadmat(path.join(path.dirname(__file__), "data", "formatted"))
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
    with tf.Session() as session:
        now = time()
        name = build_network(LAMBDA, EPSILON, LR)
        session.run(tf.global_variables_initializer())
        data = GomokuData()
        saver = tf.train.Saver(max_to_keep=10)
        root = path.join(path.dirname(__file__), "model", "policy", name)
        saver.export_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
        train_step = lambda x, y: session.run("train_step", feed_dict={"x:0": x, "y_:0": y})
        accuracy = lambda x, y: session.run("accuracy:0", feed_dict={"x:0": x, "y_:0": y})
        save = lambda i: saver.save(session, path.join(root, name), global_step=i, write_meta_graph=False)
        for i in range(MAX_STEPS):
            x_b, y_b = data.next_batch(BATCH_SIZE)
            if i % 100 == 0:
                print("step %d accuracy %g [%g sec]" % (i, accuracy(x_b, y_b), time() - now))
                now = time()
            train_step(x_b, y_b)
            if i % 1000 == 0:
                save(i)
                x_t, y_t = data.random_test(10 * BATCH_SIZE)
                print("===> validation accuracy %g (model saved)" % accuracy(x_t, y_t))
        save(MAX_STEPS)
        rates = np.zeros(shape=10, dtype=float)
        for j in range(10):
            x_t, y_t = data.random_test(10 * BATCH_SIZE)
            rates[j] = accuracy(x_t, y_t)
        print("===> validation accuracy %g (model saved)" % rates.mean())


def check(name, checkpoint=None):
    with tf.Session() as session:
        data = GomokuData()
        root = path.join(path.dirname(__file__), "model", "policy", name)
        saver = tf.train.import_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
        if checkpoint is None:
            checkpoint = int(tf.train.latest_checkpoint(root).split("-")[-1])
        saver.restore(session, path.join(root, name + "-" + str(checkpoint)))
        check_size = 10 * BATCH_SIZE
        num_checks = data.m // check_size
        accuracy = np.zeros(shape=num_checks, dtype=np.float)
        print("===> begin a total of %d check patches for %s-%d" % (num_checks, name, checkpoint))
        for i in range(num_checks):
            x_t = data.X_t[(i * check_size):min((i+1)*check_size, data.m), :]
            y_t = data.y_t[(i * check_size):min((i+1)*check_size, data.m), :]
            accuracy[i] = session.run("accuracy:0", feed_dict={"x:0": x_t, "y_:0": y_t})
            print("patch %d has accuracy %g" % (i, accuracy[i]))
        print("===> overall test accuracy %g" % np.mean(accuracy))


def resume(name, checkpoint=None):
    with tf.Session() as session:
        root = path.join(path.dirname(__file__), "model", "policy", name)
        saver = tf.train.import_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
        if checkpoint is None:
            checkpoint = int(tf.train.latest_checkpoint(root).split("-")[-1])
        saver.restore(session, path.join(root, name + "-" + str(checkpoint)))
        data = GomokuData()
        now = time()
        train_step = lambda x, y: session.run("train_step", feed_dict={"x:0": x, "y_:0": y})
        accuracy = lambda x, y: session.run("accuracy:0", feed_dict={"x:0": x, "y_:0": y})
        save = lambda i: saver.save(session, path.join(root, name), global_step=i, write_meta_graph=False)
        for i in range(checkpoint+1, MAX_STEPS):
            x_b, y_b = data.next_batch(BATCH_SIZE)
            if i % 100 == 0:
                print("step %d accuracy %g [%g sec]" % (i, accuracy(x_b, y_b), time() - now))
                now = time()
            train_step(x_b, y_b)
            if i % 1000 == 0:
                save(i)
                x_t, y_t = data.random_test(10 * BATCH_SIZE)
                print("===> validation accuracy %g (model saved)" % accuracy(x_t, y_t))

        save(MAX_STEPS)
        rates = np.zeros(shape=10, dtype=float)
        for j in range(10):
            x_t, y_t = data.random_test(10 * BATCH_SIZE)
            rates[j] = accuracy(x_t, y_t)
        print("===> validation accuracy %g (model saved)" % rates.mean())


def help():
    print("Usage: python policy.py [train/check/resume] [model] [checkpoint]")


if __name__ == "__main__":
    if 2 <= len(argv) <= 4:
        cmd = argv[1]
        if cmd == "train":
            if len(argv) != 2:
                print("modify the code to train another model")
            else:
                train()
        elif cmd == "check":
            if len(argv) == 3:
                check(argv[2], None)
            elif len(argv) == 4:
                check(argv[2], int(argv[3]))
            else:
                help()
        elif cmd == "resume":
            if len(argv) == 3:
                resume(argv[2], None)
            elif len(argv) == 4:
                resume(argv[2], int(argv[3]))
            else:
                help()
        else:
            help()
    else:
        help()
