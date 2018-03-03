""" wrapper agent for training and using tensorflow  """

import os
import numpy as np
import tensorflow as tf
from feature import diff, violate


class Agent():
    def __init__(self, sess, model_name,
        other_name=None, chkpnt=None, random=False):
        """ Constructs Tensorflow-wrapper agent
        
        Args:
            sess: tf.Session
            model_name: which model to load
            other_name: which model to store
            chkpnt: checkpoint number
            random: whether to select a random checkpoint <= chkpnt
        """
        self.sess = sess
        self.model_name = model_name
        self.meta_path = os.path.join(model_name, model_name + ".meta")
        self.saver = tf.train.import_meta_graph(self.meta_path)
        checkpoint = tf.train.latest_checkpoint(model_name)
        if chkpnt is not None:
            self.chkpnt = chkpnt
            self.saver.restore(self.sess, self.meta_path + "-" + str(chkpnt))
        elif checkpoint is not None:
            splitted = checkpoint.split("-")
            self.chkpnt = int(splitted[-1])
            if random:
                self.chkpnt = np.random.choice(self.chkpnt + 1)
                splitted[-1] = str(self.chkpnt)
            self.saver.restore(self.sess, "-".join(splitted))
        else:
            self.chkpnt = 0
            self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=999999)
        if other_name is not None:
            if not os.path.exists(other_name):
                os.makedirs(other_name)
            self.meta_path = os.path.join(other_name, other_name + ".meta")
            self.saver.export_meta_graph(self.meta_path)

    def restore(self, model_name):
        """ Restores to previous checkpoint """
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_name))

    def get_dist(self, state):
        """ Returns policy as 225 probability array """
        y_p = self.sess.run("y_p:0", feed_dict={"x_b:0": np.array([state.featurize()]), "training:0": False})[0][:225]
        y_p[(state.board != 0).reshape(225)] = 0
        assert y_p.sum() > 0
        return y_p / y_p.sum()

    def get_action(self, state, deterministic=False):
        """ Returns action based on self.get_dist """
        y_p = self.get_dist(state)
        if deterministic:
            return np.unravel_index(y_p.argmax(), dims=(15, 15))
        return np.unravel_index(np.random.choice(225, p=y_p), dims=(15, 15))

    def get_action_nonviolate(self, state):
        """ Returns action that will definitely not violate Renju """
        y_p = self.get_dist(state)
        c = np.random.choice(225, p=y_p)
        x, y = np.unravel_index(c, dims=(15, 15))
        if state.player == 1:
            while True:
                assert y_p.sum() > 0, "no more choices"
                new, old = diff(state, x, y)
                if violate(new):
                    y_p[c] = 0
                    y_p = y_p / y_p.sum()
                    c = np.random.choice(225, p=y_p)
                    x, y = np.unravel_index(c, dims=(15, 15))
                else:
                    break
        return x, y

    def accuracy(self, X, Y):
        return self.sess.run("accuracy:0", feed_dict={"x_b:0": X, "y_b:0": Y, "training:0": False})

    def loss(self, X, Y):
        return self.sess.run("loss:0", feed_dict={"x_b:0": X, "y_b:0": Y, "training:0": False})

    def step(self, X, Y):
        return self.sess.run("step", feed_dict={"x_b:0": X, "y_b:0": Y, "training:0": True})

    def save(self, global_step):
        self.saver.save(self.sess, self.meta_path, global_step=global_step, write_meta_graph=False)
