from dual_agent import DualAgent
import numpy as np


MAX_ITERS = 32
MAX_DEPTH = 16


class MCTSAgent(DualAgent):
    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs["epsilon"] or 0.05
        self.multiplier = kwargs["multiplier"] or 10
        del kwargs["epsilon"]
        del kwargs["multiplier"]
        DualAgent.__init__(self, *args, **kwargs)
        self.tree = MonteCarloTree(self.sess, self.epsilon, self.multiplier)

    def refresh(self):
        self.tree = MonteCarloTree(self.sess, self.epsilon, self.multiplier)

    def update(self, state):
        self.tree.update(state)

    def get_dist(self, state):
        self.tree.grow_tree(state)
        count = self.tree.get_visit_count()
        return count

    def get_value(self, state):
        # return a neutral representation of current value [-1, 1]
        return state.player * self.tree.get_value()

class MonteCarloTree:
    def __init__(self, sess, epsilon, multiplier):
        self.sess = sess
        self.epsilon = epsilon
        self.multiplier = multiplier
        self.root = Node()

    def grow_tree(self, state):
        for _ in range(MAX_ITERS):
            self.root.grow(self.sess, state, 0, self.epsilon, self.multiplier)

    def update(self, state):
        a = state.history[-1]
        self.root = self.root.children[np.ravel_multi_index(a, dims=(15, 15))]

    def get_value(self):
        # this value is based on the point of view of current player
        return self.root.V

    def get_visit_count(self):
        return self.root.N

    def get_action_value(self):
        return self.root.Q


class Children(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = Node()
        return dict.__getitem__(self, key)


class Node:
    def __init__(self):
        self.children = Children()
        self.W = np.zeros(shape=225, dtype=np.float32)
        self.N = np.zeros(shape=225, dtype=np.float32)
        self.P = None

    @property
    def Q(self):
        Q = np.zeros(shape=225, dtype=np.float32)
        nonzero = np.not_equal(self.N, 0)
        Q[nonzero] = self.W[nonzero] / self.N[nonzero]
        return Q

    @property
    def V(self):
        # Q is based on the point of view of current player
        # so we simply choose the maximum to be V
        return self.Q.max()

    def grow(self, sess, state, depth, epsilon, multiplier):
        if state.violate:
            # black violates the rule previously it now
            # must be white's turn; white is happy
            return 1
        if state.end:
            # the previous player wins the game
            # the current player is unhappy
            return -1
        if depth == MAX_DEPTH:
            # neural network gives a neutral representation, so we
            # need to make it in point of view of current player
            v = sess.run("v_p:0", feed_dict={"training:0": False, "x_b:0": np.array([state.featurize()])})[0]
            return state.player * v
        if self.P is None:
            P = sess.run("y_p:0", feed_dict={"training:0": False, "x_b:0": np.array([state.featurize()])})[0]
            self.P = P[:-1]
            self.P = self.P + epsilon * np.logical_and(state.adjacent() > 0, np.equal(state.board, 0)).reshape(225)
            self.P = self.P / self.P.sum()
            # if P[-1] > 0.5:
            #     # resign probability greater than 0.5
            #     return -1
        if len(state.history) == 0:
            a = np.ravel_multi_index((7, 7), dims=(15, 15))
        else:
            const = 1 + np.sqrt(np.sum(self.N))
            U = multiplier * self.Q + self.P * const / (1 + self.N)
            U[(state.board != 0).reshape(225)] = -100           # never visit illegal positions
            a = U.argmax()
        x, y = np.unravel_index(a, dims=(15, 15))
        state.move(x, y)
        v = -self.children[a].grow(sess, state, depth + 1, epsilon, multiplier)  # my point of view is opposite of opponent
        state.rewind()
        self.W[a] += v
        self.N[a] += 1
        return self.V
