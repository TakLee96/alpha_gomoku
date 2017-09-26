""" agent for training and using neural network  """

import os
import numpy as np
import tensorflow as tf


class Agent():
    
    def __init__(self, sess, model_name, other_name=None, chkpnt=None):
        self.sess = sess
        self.model_name = model_name
        self.meta_path = model_name + "/" + model_name + ".meta"
        self.saver = tf.train.import_meta_graph(self.meta_path)
        checkpoint = tf.train.latest_checkpoint(model_name)
        if chkpnt is not None:
            self.saver.restore(self.sess, model_name + "/" + model_name + ".meta-" + str(chkpnt))
        elif checkpoint is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, checkpoint)
        self.saver = tf.train.Saver(max_to_keep=999999)
        if other_name is not None:
            if not os.path.exists(other_name):
                os.makedirs(other_name)
            self.meta_path = other_name + "/" + other_name + ".meta"
            self.saver.export_meta_graph(self.meta_path)

    def get_dist(self, state):
        y_p = self.sess.run("y_p:0", feed_dict={"x_b:0": np.array([state.featurize()])})[0]
        y_p[(state.board != 0).reshape(225)] = 0
        return y_p / y_p.sum()

    def get_action(self, state):
        y_p = self.get_dist(state)
        x, y = np.unravel_index(np.random.choice(225, p=y_p), dims=(15, 15))
        assert state.board[x, y] == 0, "total prob %f" % y_p.sum()
        return x, y

    def accuracy(self, X, Y):
        return self.sess.run("accuracy:0", feed_dict={"x_b:0": X, "y_b:0": Y})

    def loss(self, X, Y):
        return self.sess.run("loss:0", feed_dict={"x_b:0": X, "y_b:0": Y})

    def pg_loss(self, X, Y, A):
        return self.sess.run("pg_loss:0", feed_dict={"x_b:0": X, "y_b:0": Y, "adv_b:0": A})

    def step(self, X, Y):
        return self.sess.run("step", feed_dict={"x_b:0": X, "y_b:0": Y})

    def pg_step(self, X, Y, A):
        return self.sess.run("pg_step", feed_dict={"x_b:0": X, "y_b:0": Y, "adv_b:0": A})

    def save(self, global_step):
        self.saver.save(self.sess, self.meta_path, global_step=global_step, write_meta_graph=False)
