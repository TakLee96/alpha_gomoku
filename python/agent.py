""" agent for training and using neural network  """

import numpy as np
import tensorflow as tf


class Agent():
    
    def __init__(self, sess, model_folder, model_name):
        self.sess = sess
        self.saver = tf.train.import_meta_graph(model_folder + "/" + model_name + ".meta")
        checkpoint = tf.train.latest_checkpoint(model_folder)
        if checkpoint is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, checkpoint)
        self.saver = tf.train.Saver(max_to_keep=999999)

    def get_dist(self, state):
        y_p = self.sess.run("y_p:0", feed_dict={"x_b:0": np.array([state.featurize()])})[0]
        y_p[(state.board != 0).reshape(225)] = 0
        return y_p / y_p.sum()

    def get_action(self, state):
        y_p = self.get_dist(state)
        return np.unravel_index(np.random.choice(225, p=y_p), dims=(15, 15))

    def loss(self, X, Y):
        return self.sess.run("loss:0", feed_dict={"x_b:0": X, "y_b:0": Y})

    def step(self, X, Y):
        return self.sess.run("step", feed_dict={"x_b:0": X, "y_b:0": Y})

    def save(self, global_step):
        self.saver.save(self.sess, model_folder + "/" + model_name + ".meta",
            global_step=global_step, write_meta_graph=False)
