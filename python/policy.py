""" neural network training """
import numpy as np
import tensorflow as tf
from os import path
from sys import argv
from time import time
from scipy.io import loadmat
from network.policy.kunet import build_network


""" Hyperparameters """
LR = 1e-3
LAMBDA = 1e-8
EPSILON = 1e-7
MAX_STEPS = 30000
BATCH_SIZE = 150


class GomokuData():
    @staticmethod
    def _one_hot(y):
        n = len(y)
        y_h = np.zeros(shape=(n, 225), dtype=np.float32)
        for i in range(n):
            y_h[i, y[i]] = 1.0
        return y_h

    def __init__(self, which):
        mat = loadmat(path.join(path.dirname(__file__), "data", "minimax_" + which))
        self.X_t = mat["X_t"].astype(np.float32)
        self.y_t = self._one_hot(mat["y_t"][0])
        self.f_t = mat["f_t"][0]
        self.X_v = mat["X_v"].astype(np.float32)
        self.y_v = self._one_hot(mat["y_v"][0])
        self.f_v = mat["f_v"][0]
        self.n, _, _, _ = self.X_t.shape
        self.m, _, _, _ = self.X_v.shape
        self.cursor = self.n

    def reset(self):
        self.cursor = 0

    def next_batch(self, size):
        if self.cursor + size > self.n:
            self.cursor = 0
            ordering = np.random.permutation(self.n)
            self.X_t = self.X_t[ordering, :]
            self.y_t = self.y_t[ordering, :]
            self.f_t = self.f_t[ordering]
        X_b = self.X_t[self.cursor:(self.cursor + size), :]
        y_b = self.y_t[self.cursor:(self.cursor + size), :]
        f_b = self.f_t[self.cursor:(self.cursor + size)]
        self.cursor += size
        return X_b, y_b, f_b

    def random_test(self, size):
        which = np.random.choice(self.m, size, replace=False)
        X_b = self.X_v[which, :]
        y_b = self.y_v[which, :]
        f_b = self.f_v[which]
        return X_b, y_b, f_b


def train(which):
    with tf.Session() as session:
        now = time()
        name = build_network(LAMBDA, EPSILON, LR) + "-" + which
        session.run(tf.global_variables_initializer())
        data = GomokuData(which)
        saver = tf.train.Saver(max_to_keep=99999999)
        root = path.join(path.dirname(__file__), "model", "policy", name)
        saver.export_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
        train_step = lambda x, y, f: session.run("train_step", feed_dict={"x:0": x, "y:0": y, "f:0": f})
        likelihood = lambda x, y, f: session.run("likelihood:0", feed_dict={"x:0": x, "y:0": y, "f:0": f})
        save = lambda i: saver.save(session, path.join(root, name), global_step=i, write_meta_graph=False)
        for i in range(MAX_STEPS):
            x_b, y_b, f_b = data.next_batch(BATCH_SIZE)
            if i % 100 == 0:
                print("step %d likelihood %g [%g sec]" % (i, likelihood(x_b, y_b, f_b), time() - now))
                now = time()
            train_step(x_b, y_b, f_b)
            if i % 1000 == 0:
                save(i)
                print("===> 10 check validation likelihood %g (model saved)" %
                    np.mean([likelihood(*data.random_test(10 * BATCH_SIZE)) for _ in range(10)]))
        save(MAX_STEPS)
        print("===> 10 check validation likelihood %g (model saved)" %
            np.mean([likelihood(*data.random_test(10 * BATCH_SIZE)) for _ in range(10)]))


def check(which, name, checkpoint=None):
    with tf.Session() as session:
        data = GomokuData(which)
        name = name + "-" + which
        root = path.join(path.dirname(__file__), "model", "policy", name)
        saver = tf.train.import_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
        if checkpoint is None:
            checkpoint = int(tf.train.latest_checkpoint(root).split("-")[-1])
        saver.restore(session, path.join(root, name + "-" + str(checkpoint)))
        check_size = 10 * BATCH_SIZE
        num_checks = data.m // check_size
        if num_checks > 1:
            likelihood = np.zeros(shape=num_checks, dtype=np.float)
            print("===> begin a total of %d check patches for %s-%d" % (num_checks, name, checkpoint))
            for i in range(num_checks):
                x_t = data.X_v[(i * check_size):min((i+1)*check_size, data.m), :]
                y_t = data.y_v[(i * check_size):min((i+1)*check_size, data.m), :]
                f_t = data.f_v[(i * check_size):min((i+1)*check_size, data.m)]
                likelihood[i] = session.run("likelihood:0", feed_dict={"x:0": x_t, "y:0": y_t, "f:0": f_t})
                print("patch %d has likelihood %g" % (i, likelihood[i]))
            print("===> overall test likelihood %g" % np.mean(likelihood))
        else:
            print("===> test likelihood %g" % session.run("likelihood:0", feed_dict={"x:0": data.X_v, "y:0": data.y_v, "f:0": data.f_v}))


def resume(which, name, checkpoint=None):
    with tf.Session() as session:
        name = name + "-" + which
        root = path.join(path.dirname(__file__), "model", "policy", name)
        saver = tf.train.import_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
        if checkpoint is None:
            checkpoint = int(tf.train.latest_checkpoint(root).split("-")[-1])
        saver.restore(session, path.join(root, name + "-" + str(checkpoint)))
        saver = tf.train.Saver(max_to_keep=99999999)
        data = GomokuData(which)
        now = time()
        train_step = lambda x, y, f: session.run("train_step", feed_dict={"x:0": x, "y:0": y, "f:0": f})
        likelihood = lambda x, y, f: session.run("likelihood:0", feed_dict={"x:0": x, "y:0": y, "f:0": f})
        save = lambda i: saver.save(session, path.join(root, name), global_step=i, write_meta_graph=False)
        for i in range(checkpoint+1, MAX_STEPS):
            x_b, y_b, f_b = data.next_batch(BATCH_SIZE)
            if i % 100 == 0:
                print("step %d likelihood %g [%g sec]" % (i, likelihood(x_b, y_b, f_b), time() - now))
                now = time()
            train_step(x_b, y_b, f_b)
            if i % 1000 == 0:
                save(i)
                print("===> 10 check validation likelihood %g (model saved)" %
                    np.mean([likelihood(*data.random_test(10 * BATCH_SIZE)) for _ in range(10)]))
        save(MAX_STEPS)
        print("===> 10 check validation likelihood %g (model saved)" %
            np.mean([likelihood(*data.random_test(10 * BATCH_SIZE)) for _ in range(10)]))


def help():
    print("Usage: python policy.py [train/check/resume] [black/white] [model] [checkpoint]")


if __name__ == "__main__":
    if 3 <= len(argv) <= 5:
        cmd = argv[1]
        which = argv[2]
        if which not in ("black", "white"):
            help()
        elif cmd == "train":
            if len(argv) != 3:
                print("modify the code to train another model")
            else:
                train(argv[2])
        elif cmd == "check":
            if len(argv) == 4:
                check(argv[2], argv[3], None)
            elif len(argv) == 5:
                check(argv[2], argv[3], int(argv[4]))
            else:
                help()
        elif cmd == "resume":
            if len(argv) == 4:
                resume(argv[2], argv[3], None)
            elif len(argv) == 5:
                resume(argv[2], argv[3], int(argv[4]))
            else:
                help()
        else:
            help()
    else:
        help()
