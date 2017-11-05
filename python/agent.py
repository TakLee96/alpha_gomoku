""" agent for training and using neural network  """

import os
import numpy as np
import tensorflow as tf

changes = [
    lambda b: b,
    lambda b: np.transpose(b, (1, 0, 2)),
    lambda b: b[:, ::-1, :],
    lambda b: b[::-1, :, :],
    lambda b: b[::-1, ::-1, :],
    lambda b: np.transpose(b, (1, 0, 2))[:, ::-1, :],
    lambda b: np.transpose(b, (1, 0, 2))[::-1, :, :],
    lambda b: np.transpose(b, (1, 0, 2))[::-1, ::-1, :],
]
reverses = [
    lambda p: p,
    lambda p: p.T,
    lambda p: p[:, ::-1],
    lambda p: p[::-1, :],
    lambda p: p[::-1, ::-1],
    lambda p: p[:, ::-1].T,
    lambda p: p[::-1, :].T,
    lambda p: p[::-1, ::-1].T,
]

class Agent():
    
    def __init__(self, sess, model_name,
        other_name=None, chkpnt=None, ensemble=False, random=False):
        self.sess = sess
        self.ensemble = ensemble
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
            self.meta_path = other_name + "/" + other_name + ".meta"
            self.saver.export_meta_graph(self.meta_path)

    def get_dist_ensemble(self, state):
        states = []
        for change in changes:
            states.append(change(state.featurize()))
        dists = self.sess.run("y_p:0", feed_dict={"x_b:0": np.array(states)})
        dist = np.zeros(shape=225, dtype=np.float32)
        for i in range(len(reverses)):
            y_p = dists[i].reshape((15, 15))
            dist += reverses[i](y_p / y_p.sum()).reshape(225)
        dist[(state.board != 0).reshape(225)] = 0
        assert dist.sum() > 0
        return dist / dist.sum()

    def get_dist(self, state):
        y_p = self.sess.run("y_p:0", feed_dict={"x_b:0": np.array([state.featurize()])})[0]
        y_p[(state.board != 0).reshape(225)] = 0
        assert y_p.sum() > 0
        return y_p / y_p.sum()

    def get_action(self, state, deterministic=False):
        if self.ensemble:
            y_p = self.get_dist_ensemble(state)
        else:
            y_p = self.get_dist(state)
        if deterministic:
            return np.unravel_index(y_p.argmax(), dims=(15, 15))
        c = np.random.choice(225, p=y_p)
        x, y = np.unravel_index(c, dims=(15, 15))
        assert state.board[x, y] == 0, "total prob %f that prob %f" % (y_p.sum(), y_p[c])
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
