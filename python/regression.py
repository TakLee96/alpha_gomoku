""" neural network training """
import numpy as np
import tensorflow as tf
from os import path
from sys import argv
from time import time
from scipy.io import loadmat
from network.value.qbtnet import build_network


""" Hyperparameters """
LR = 1e-3
LAMBDA = 1e-3
MAX_STEPS = 30000
BATCH_SIZE = 500


class GameData():
    def __init__(self, which):
        mat = loadmat(path.join(path.dirname(__file__), "data", which))
        self.X_t = mat["X_t"]
        self.y_t = mat["y_t"][0]
        self.X_v = mat["X_v"]
        self.y_v = mat["y_v"][0]
        self.n, _ = self.X_t.shape
        self.m, _ = self.X_v.shape
        self.cursor = self.n

    def next_batch(self, size):
        if self.cursor + size > self.n:
            self.cursor = 0
            ordering = np.random.permutation(self.n)
            self.X_t = self.X_t[ordering, :]
            self.y_t = self.y_t[ordering]
        X_b = self.X_t[self.cursor:(self.cursor + size), :]
        y_b = self.y_t[self.cursor:(self.cursor + size)]
        self.cursor += size
        return X_b, y_b

    def random_test(self, size):
        which = np.random.choice(self.m, size, replace=False)
        X_b = self.X_v[which, :]
        y_b = self.y_v[which]
        return X_b, y_b


def train(which):
    with tf.Session() as session:
        name = build_network(LAMBDA, LR) + "-" + which
        session.run(tf.global_variables_initializer())
        data = GameData(which)
        root = path.join(path.dirname(__file__), "model", name)
        saver = tf.train.Saver(max_to_keep=10)
        saver.export_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
        train_step = lambda x, y: session.run("train_step", feed_dict={"x:0": x, "y_:0": y.reshape((len(y), 1))})
        loss = lambda x, y: session.run("loss:0", feed_dict={"x:0": x, "y_:0": y.reshape((len(y), 1))})
        save = lambda i: saver.save(session, path.join(root, name), global_step=i, write_meta_graph=False)
        now = time()
        for i in range(MAX_STEPS):
            x_b, y_b = data.next_batch(BATCH_SIZE)
            if i % 100 == 0:
                print("step %d loss %g [%g sec]" % (i, loss(x_b, y_b), time() - now))
                now = time()
            train_step(x_b, y_b)
            if i % 1000 == 0:
                save(i)
                print("===> validation loss %g (model saved)" % loss(data.X_v, data.y_v))
        save(MAX_STEPS)
        print("===> validation loss %g (model saved)" % loss(data.X_v, data.y_v))


def check(which, name, checkpoint=None):
    with tf.Session() as session:
        data = GameData(which)
        root = path.join(path.dirname(__file__), "model", name)
        saver = tf.train.import_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
        if checkpoint is None:
            checkpoint = int(tf.train.latest_checkpoint(root).split("-")[-1])
        saver.restore(session, path.join(root, name + "-" + str(checkpoint)))
        loss = session.run("loss:0", feed_dict={"x:0": data.X_v, "y_:0": y_v})
        print("===> validation loss %g (model saved)" % loss)


def resume(which, name, checkpoint=None):
    with tf.Session() as session:
        root = path.join(path.dirname(__file__), "model", name)
        saver = tf.train.import_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
        if checkpoint is None:
            checkpoint = int(tf.train.latest_checkpoint(root).split("-")[-1])
        saver.restore(session, path.join(root, name + "-" + str(checkpoint)))
        data = GameData(which)
        train_step = lambda x, y: session.run("train_step", feed_dict={"x:0": x, "y_:0": y})
        loss = lambda x, y: session.run("loss:0", feed_dict={"x:0": x, "y_:0": y})
        save = lambda i: saver.save(session, path.join(root, name), global_step=i, write_meta_graph=False)
        now = time()
        for i in range(checkpoint+1, MAX_STEPS):
            x_b, y_b = data.next_batch(BATCH_SIZE)
            if i % 100 == 0:
                print("step %d loss %g [%g sec]" % (i, loss(x_b, y_b), time() - now))
                now = time()
            train_step(x_b, y_b)
            if i % 1000 == 0:
                save(i)
                print("===> validation loss %g (model saved)" % loss(data.X_v, data.y_v))
        save(MAX_STEPS)
        print("===> validation loss %g (model saved)" % loss(data.X_v, data.y_v))


def help():
    print("Usage: python regression.py [train/check/resume] [black/white] [model] [checkpoint]")


if __name__ == "__main__":
    if 2 <= len(argv) <= 5:
        cmd = argv[1]
        if cmd == "train":
            if len(argv) != 3:
                print("Usage: python regression.py train [black/white]")
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
