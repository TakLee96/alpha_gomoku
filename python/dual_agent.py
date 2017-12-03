import numpy as np
import tensorflow as tf
from agent import Agent


class DualAgent(Agent):
    def get_dist(self, state):
        y_p, v_p = self.sess.run(["y_p:0", "v_p:0"], feed_dict={"training:0": False, "x_b:0": np.array([state.featurize()])})
        y_p, v_p = y_p[0][:-1], v_p[0]
        y_p[(state.board != 0).reshape(225)] = 0
        assert y_p.sum() > 0, "resign"
        #print("value: %f" % v_p)
        return y_p / y_p.sum()

    def accuracy(self, X, Y):
        return self.sess.run("accuracy:0", feed_dict={"x_b:0": X, "y_b:0": Y, "training:0": False})

    def loss(self, X, Y, V):
        return self.sess.run(["policy_loss:0", "value_loss:0", "regularization_loss:0", "loss:0"], feed_dict={"x_b:0": X, "y_b:0": Y, "v_b:0": V, "training:0": False})

    def step(self, X, Y, V):
        return self.sess.run("step", feed_dict={"x_b:0": X, "y_b:0": Y, "v_b:0": V, "training:0": True})
