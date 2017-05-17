""" Convolutional Neural Network for Policy and Value """
import numpy as np
import tensorflow as tf
from time import time
from scipy.io import loadmat

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

def build_network():
    x = tf.placeholder(tf.float32, shape=[None, 225], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 225], name="y_")
    is_training = tf.placeholder(tf.bool, name="is_training")
    board = tf.reshape(x, [-1, 15, 15, 1])

    conv_layer_1 = tf.contrib.layers.conv2d(
        inputs=board, num_outputs=64, kernel_size=7, stride=1, padding="VALID",
        activation_fn=None, biases_initializer=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    norm_layer_1 = tf.contrib.layers.batch_norm(
        inputs=conv_layer_1, decay=0.9, center=True, scale=True, epsilon=EPSILON,
        is_training=is_training, activation_fn=tf.nn.relu, trainable=True)

    conv_layer_2 = tf.contrib.layers.conv2d(
        inputs=norm_layer_1, num_outputs=16, kernel_size=1, stride=1, padding="VALID",
        activation_fn=None, biases_initializer=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    norm_layer_2 = tf.contrib.layers.batch_norm(
        inputs=conv_layer_2, decay=0.9, center=True, scale=True, epsilon=EPSILON,
        is_training=is_training, activation_fn=tf.nn.relu, trainable=True)

    flatten = tf.reshape(norm_layer_2, [-1, 9 * 9 * 16])
    fc_layer_3 = tf.contrib.layers.fully_connected(
        inputs=flatten, num_outputs=256, activation_fn=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))
    norm_layer_3 = tf.contrib.layers.batch_norm(
        inputs=fc_layer_3, decay=0.9, center=True, scale=True, epsilon=EPSILON,
        is_training=is_training, activation_fn=tf.nn.relu, trainable=True)

    y = tf.contrib.layers.fully_connected(
        inputs=norm_layer_3, num_outputs=225, activation_fn=None, trainable=True,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=LAMBDA))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name="loss")
    tf.train.AdamOptimizer(LR).minimize(loss, name="train_step")
    tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32), name="accuracy")

def train():
    with tf.Session() as sess:
        now = time()
        build_network()
        sess.run(tf.global_variables_initializer())
        data = GomokuData()
        s = tf.train.Saver()
        s.export_meta_graph("model/version_one.meta", clear_devices=True)

        train_step = lambda x, y: sess.run("train_step", feed_dict={"x:0": x, "y_:0": y, "is_training:0": True})
        accuracy = lambda x, y: sess.run("accuracy:0", feed_dict={"x:0": x, "y_:0": y, "is_training:0": False})
        save = lambda i: s.save(sess, "model/version_one", global_step=i, write_meta_graph=False)

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

train()
