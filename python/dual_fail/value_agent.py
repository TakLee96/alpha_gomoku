""" agent for training and using neural network  """

import os
import numpy as np
import tensorflow as tf


wikipedia = [
    "-xxo", "-x-xo", "-oox", "-o-ox",
    "-x-x-", "-xx-", "-o-o-", "-oo-",
    "-x-xxo", "-xxxo", "-o-oox", "-ooox",
    "-x-xx-", "-xxx-", "-o-oo-", "-ooo-",
    "-xxxx-", "-xxxxo", "-oooo-", "-oooox",
    "four-o", "four-x", "win-o", "win-x", "violate"
]
wikipedia = { k: i for i, k in enumerate(wikipedia) }


class ValueAgent():
    
    def __init__(self, sess, model_name,
        other_name=None, chkpnt=None):
        self.sess = sess
        self.model_name = model_name
        self.meta_path = model_name + "/" + model_name + ".meta"
        self.saver = tf.train.import_meta_graph(self.meta_path)
        checkpoint = tf.train.latest_checkpoint(model_name)
        if chkpnt is not None:
            self.chkpnt = chkpnt
            self.saver.restore(self.sess, model_name + "/" + model_name + ".meta-" + str(chkpnt))
        elif checkpoint is not None:
            splitted = checkpoint.split("-")
            self.chkpnt = int(splitted[-1])
            self.saver.restore(self.sess, "-".join(splitted))
        else:
            self.chkpnt = 0
            self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=999999)
        if other_name is not None:
            if not os.path.exists(other_name):
                os.makedirs(other_name)
            self.meta_path = other_name + "/" + other_name + ".meta"
            self.saver.export_meta_graph(self.meta_path)

    def restore(self, model_name):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_name))

    def step(self, X, V):
        return self.sess.run("step", feed_dict={"x_b:0": X, "v_b:0": V, "training:0": True})

    def reg_loss(self, X, V):
        return self.sess.run("regularization_loss:0", feed_dict={"x_b:0": X, "v_b:0": V, "training:0": False})

    def l2_loss(self, X, V):
        return self.sess.run("value_loss:0", feed_dict={"x_b:0": X, "v_b:0": V, "training:0": False})

    def save(self, global_step):
        self.saver.save(self.sess, self.meta_path, global_step=global_step, write_meta_graph=False)

    def value(self, state):
        # features = np.zeros(shape=2*len(wikipedia), dtype=np.float32)
        # for feature in state.features.keys():
        #     if state.features[feature] > 0:
        #         assert feature in wikipedia, "unknown " + feature
        #         location = wikipedia[feature]
        #         if state.player == 1:
        #             location = location
        #         else:
        #             location = location + len(wikipedia)
        #         features[location] = state.features[feature]
        # return self.sess.run("v_p:0", feed_dict={"x_b:0": np.array([features]), "training:0": False})[0]
        return self.sess.run("v_p:0", feed_dict={"x_b:0": np.array([state.featurize()]), "training:0": False})[0]