""" alpha-beta minimax agent """

import random
import numpy as np
import data_util as util
from agent import Agent
from feature import diff
from scipy.signal import convolve2d as conv2d

INFINITY = 1000
DANGER = 100
DISCOUNT = 0.99

class BlackOps:
    @staticmethod
    def get_op_live_four(f):
        return f["-xxxx-"]
    @staticmethod
    def get_op_four(f):
        return f["four-x"] + f["-xxxxo"]
    @staticmethod
    def get_op_three(f):
        return f["-x-xx-"] + f["-xxx-"]
    @staticmethod
    def get_op_potential(f):
        return f["-xx-"] + f["-x-x-"] + f["-x-xxo"] + f["-xxxo"]
    @staticmethod
    def get_my_live_four(f):
        return f["-oooo-"]
    @staticmethod
    def get_my_four(f):
        return f["four-o"] + f["-oooox"]
    @staticmethod
    def get_my_three(f):
        return f["-o-oo-"] + f["-ooo-"]
    @staticmethod
    def get_my_potential(f):
        return f["-oo-"] + f["-o-o-"] + f["-o-oox"] + f["-ooox"]
    

class WhiteOps:
    @staticmethod
    def get_my_live_four(f):
        return f["-xxxx-"]
    @staticmethod
    def get_my_four(f):
        return f["four-x"] + f["-xxxxo"]
    @staticmethod
    def get_my_three(f):
        return f["-x-xx-"] + f["-xxx-"]
    @staticmethod
    def get_my_potential(f):
        return f["-xx-"] + f["-x-x-"] + f["-x-xxo"] + f["-xxxo"]
    @staticmethod
    def get_op_live_four(f):
        return f["-oooo-"]
    @staticmethod
    def get_op_four(f):
        return f["four-o"] + f["-oooox"]
    @staticmethod
    def get_op_three(f):
        return f["-o-oo-"] + f["-ooo-"]
    @staticmethod
    def get_op_potential(f):
        return f["-oo-"] + f["-o-o-"] + f["-o-oox"] + f["-ooox"]


class MinimaxNetworkAgent(Agent):
    def __init__(self, *args, **kwargs):
        self.max_depth = 8
        if "max_depth" in kwargs:
            self.max_depth = kwargs["max_depth"] or self.max_depth
            del kwargs["max_depth"]
        self.max_width = 3
        if "max_width" in kwargs:
            self.max_width = kwargs["max_width"] or self.max_width
            del kwargs["max_width"]
        Agent.__init__(self, *args, **kwargs)
        self.ops = [None, BlackOps(), WhiteOps()]

    def random_action(self, state):
        adjacent = state.adjacent()
        prob = np.logical_and(state.board == 0, adjacent > 0).astype(float)
        for x in range(15):
            for y in range(15):
                if prob[x, y] > 0:
                    new, old = diff(state, x, y)
                    if ("violate" in new or new["-o-oo-"] + new["-ooo-"] >= 2 or
                        new["four-o"] + new["-oooo-"] + new["-oooox"] >= 2):
                        prob[x, y] = 0
        prob = (prob / prob.sum()).reshape(225)
        return np.unravel_index(np.random.choice(225, p=prob), dims=(15, 15))

    def _policy(self, state):
        if len(state.history) == 0:
            return [(7, 7)]
        actions = []
        if len(state.history) == 1:
            x, y = state.history[0]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx != 0 or dy != 0:
                        if 0 <= x + dx <= 14 and 0 <= y + dy <= 14:
                            actions.append((x + dx, y + dy))
        else:
            adjacent = state.adjacent()
            ops = self.ops[state.player]
            in_lose_danger = ops.get_op_live_four(state.features) > 0
            in_four_danger = ops.get_op_four(state.features) > 0
            in_three_danger = ops.get_op_three(state.features) > 0
            for x in range(15):
                for y in range(15):
                    if adjacent[x, y] > 0 and state.board[x, y] == 0:
                        new, old = diff(state, x, y)
                        if state.player == -1 or ("violate" not in new and
                            new["-o-oo-"] + new["-ooo-"] < 2 and
                            new["four-o"] + new["-oooo-"] + new["-oooox"] < 2):
                            if new["win-o"] > 0 or new["win-x"] > 0:
                                return [(x, y)]
                            elif not in_lose_danger:
                                if in_four_danger:
                                    if ops.get_op_four(state.features) == ops.get_op_four(old):
                                        actions.append((x, y))
                                elif ops.get_my_live_four(new) > 0:
                                    return [(x, y)]
                                elif in_three_danger:
                                    if ops.get_my_four(new) > 0:
                                        actions.append((x, y))
                                    elif ops.get_op_three(state.features) == ops.get_op_three(old):
                                        actions.append((x, y))
                                elif len(new) + len(old) > 0:
                                    actions.append((x, y))
            # if in_four_danger or in_three_danger:
            #     return actions
        y = Agent.get_dist(self, state)
        return sorted(actions, key=lambda t: y[np.ravel_multi_index(t, dims=(15, 15))], reverse=True)[:min(self.max_width, len(actions))]

    def _value(self, state):
        ops = self.ops[state.player]
        if ops.get_my_live_four(state.features) > 0 or ops.get_my_four(state.features) > 0:
            return state.player * INFINITY
        if ops.get_op_live_four(state.features) > 0:
            return -state.player * INFINITY
        if ops.get_op_four(state.features) > 0:
            if ops.get_op_four(state.features) > 1:
                return -state.player * DANGER
            if ops.get_op_four(state.features) + ops.get_op_three(state.features) > 1:
                return -state.player * DANGER
        if ops.get_my_three(state.features) > 0:
            return state.player * INFINITY
        if ops.get_op_three(state.features) > 1:
            return -state.player * INFINITY
        return state.player * (ops.get_my_potential(state.features) * 2.0
            - ops.get_op_potential(state.features) * 1.0
            - ops.get_op_three(state.features) * 2.0
            - ops.get_op_four(state.features) * 3.0)

    def _max_value(self, state, alpha, beta, depth, hist, store):
        if state.end:
            return state.player * INFINITY
        if depth == self.max_depth:
            return self._value(state)
        frozen = frozenset(hist)
        if frozen in store:
            return store[frozen]
        actions = self._policy(state)
        if len(actions) == 0:
            return -INFINITY
        who = state.player
        max_value = -INFINITY
        for x, y in actions:
            hist.add(who * np.ravel_multi_index((x, y), dims=(15, 15)))
            state.move(x, y)
            max_value = max(max_value,
                DISCOUNT * self._min_value(state, alpha, beta, depth + 1, hist, store))
            if max_value > beta:
                hist.remove(who * np.ravel_multi_index((x, y), dims=(15, 15)))
                state.rewind()
                store[frozenset(hist)] = max_value
                return max_value
            alpha = max(alpha, max_value)
            hist.remove(who * np.ravel_multi_index((x, y), dims=(15, 15)))
            state.rewind()
        store[frozenset(hist)] = max_value
        return max_value

    def _min_value(self, state, alpha, beta, depth, hist, store):
        if state.end:
            return state.player * INFINITY
        if depth == self.max_depth:
            return self._value(state)
        frozen = frozenset(hist)
        if frozen in store:
            return store[frozen]
        actions = self._policy(state)
        if len(actions) == 0:
            return INFINITY
        who = state.player
        min_value = INFINITY
        for x, y in actions:
            hist.add(who * np.ravel_multi_index((x, y), dims=(15, 15)))
            state.move(x, y)
            min_value = min(min_value,
                DISCOUNT * self._max_value(state, alpha, beta, depth + 1, hist, store))
            if min_value < alpha:
                hist.remove(who * np.ravel_multi_index((x, y), dims=(15, 15)))
                state.rewind()
                store[frozenset(hist)] = min_value
                return min_value
            beta = min(beta, min_value)
            hist.remove(who * np.ravel_multi_index((x, y), dims=(15, 15)))
            state.rewind()
        store[frozenset(hist)] = min_value
        return min_value

    def get_action(self, state):
        dist = self._get_dist(state)
        prob = np.array(list(map(lambda t: t[2], dist)))
        choice = np.random.choice(prob.shape[0], p=prob)
        x, y, _ = dist[choice]
        return x, y

    def _get_dist(self, state):
        score = self.get_score(state)
        prob = np.array(list(map(lambda t: state.player * t[2], score)))
        prob = np.exp(prob - prob.max())
        prob = prob / prob.sum()
        return list(map(lambda i: (score[i][0], score[i][1], prob[i]), range(len(score))))

    def get_dist(self, state):
        dist = self._get_dist(state)
        return util.dist_to_prob(dist)

    def get_dist_ensemble(self, state):
        return self.get_dist(state)

    def get_score_dist(self, state):
        return util.dist_to_prob(list(map(lambda t: (t[0], t[1], state.player * t[2]), self.get_score(state))))

    def get_score(self, state):
        actions = self._policy(state)
        if len(actions) == 0:
            # return [(-1, -1, 0)]
            x, y = self.random_action(state)
            return [(x, y, -1)]
        if len(actions) == 1:
            x, y = actions[0]
            return [(x, y, -1)]
        who = state.player
        store = dict()
        def evaluate(action):
            x, y = action
            hist = { who * np.ravel_multi_index((x, y), dims=(15, 15)) }
            state.move(x, y)
            if who == 1:
                val = self._min_value(state, -INFINITY, INFINITY, 1, hist, store)
            else:
                val = self._max_value(state, -INFINITY, INFINITY, 1, hist, store)
            state.rewind()
            return x, y, val
        return list(map(evaluate, actions))
