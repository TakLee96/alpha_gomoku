from agent import Agent
from feature import diff
from functools import reduce
import numpy as np


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
Ops = [None, BlackOps(), WhiteOps()]

MAX_ITERS = 64
MAX_DEPTH = 10
INFINITY  = 10
EPSILON   = 0.05
MULTIPLE  = 0.5


def tuple_to_int(t):
    return np.ravel_multi_index(t, dims=(15, 15))

def int_to_tuple(i):
    return np.unravel_index(i, dims=(15, 15))

def get_reasonable_actions(state):
    if len(state.history) == 0:
        return [(7, 7)]
    elif len(state.history) == 1:
        x, y = state.history[0]
        actions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    if 0 <= x + dx <= 14 and 0 <= y + dy <= 14:
                        actions.append((x + dx, y + dy))
        return actions
    adjacent = state.adjacent()
    actions = []
    ops = Ops[state.player]
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
    return actions


class MCTSMinimaxAgent():
    def __init__(self, *args, **kwargs):
        self.agent = Agent(*args, **kwargs)
        self.root = MCTSNode()

    def get_action(self, state):
        for _ in range(MAX_ITERS):
            self.root.grow(self, state, 0)
        if self.root.p is None or len(self.root.p) == 0:
            return (-1, -1)
        if self.root.deterministic and self.root.v == -INFINITY:
            return (-1, -1)
        best_score  = -INFINITY
        best_action = None
        for action, (prob, node) in self.root.p.items():
            if node.n > 0:
                if node.deterministic and -node.v == INFINITY:
                    return action
                score = -node.v + MULTIPLE * np.sqrt(node.n)
                if score > best_score:
                    best_score  = score
                    best_action = action
        assert best_action is not None, "no available actions"
        return best_action

    def get_dist(self, state):
        for _ in range(MAX_ITERS):
            self.root.grow(self, state, 0)
        assert self.root.p is not None and len(self.root.p) > 0, "no available actions"
        print(self.root)
        p = np.zeros(shape=225, dtype=np.float32)
        for action, (prob, node) in self.root.p.items():
            if node.n > 0:
                p[tuple_to_int(action)] = np.exp(-node.v + MULTIPLE * np.sqrt(node.n))
        if p.sum() > 0:
            p = p / p.sum()
            return p
        else:
            return p

    def refresh(self):
        self.root = MCTSNode()

    def update(self, state):
        if self.root.p is not None and state.history[-1] in self.root.p:
            self.root = self.root.p[state.history[-1]][1]
        else:
            self.root = MCTSNode()

    def rollout(self, state):
        t = state.copy()
        while not t.end and len(t.history) < min(len(state.history) + 20, 225):
            t.move(*self.agent.get_safe_action(t))
        if t.end:
            return state.player * t.player
        return 0

    def get_nn_dist(self, state):
        return self.agent.get_dist(state)


class MCTSNode:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.deterministic = False
        self.is_leaf = False
        self.p = None

    def __str__(self):
        result = "Node(n=%d, w=%.02f, deterministic=%r, is_leaf=%r)" % \
            (self.n, self.w, self.deterministic, self.is_leaf)
        for action, (prob, node) in self.p.items():
            if node.n > 0:
                result += "\n    %r => p: %.02f v: %+.02f n: %d leaf: %r deter: %r" % (action, prob, -node.v, node.n, node.is_leaf, node.deterministic)
        return result

    @property
    def v(self):
        assert self.n != 0, "getting value from unvisited node"
        if self.is_leaf:
            return self.w / self.n
        else:
            return self.w

    def grow(self, agent, state, depth):
        if self.deterministic:
            return
        if state.violate:
            self.n = 1
            self.w = INFINITY
            self.deterministic = True
            self.is_leaf = True
        elif state.end:
            self.n = 1
            self.w = -INFINITY
            self.deterministic = True
            self.is_leaf = True
        elif depth == MAX_DEPTH:
            self.n += 1
            self.w += agent.rollout(state)
            self.deterministic = False
            self.is_leaf = True
        else:
            if self.p is None:
                reasonable_actions = get_reasonable_actions(state)
                if len(reasonable_actions) == 0:
                    self.n = 1
                    self.w = -INFINITY
                    self.deterministic = True
                    self.is_leaf = True
                    return
                p = agent.get_nn_dist(state)
                rescale = 1 + len(reasonable_actions) * EPSILON
                self.p = { a: ((p[tuple_to_int(a)] + EPSILON) / rescale,
                    MCTSNode()) for a in reasonable_actions }
            if self.is_leaf:
                self.is_leaf = False
                self.n = 0
                self.w = 0
            which = None
            best_score = -INFINITY
            for action, (prob, node) in self.p.items():
                score = -INFINITY
                if node.n == 0:
                    score = MULTIPLE * prob * (1 + np.sqrt(self.n)) / (1 + node.n)
                elif not node.deterministic:
                    score = -node.v + MULTIPLE * prob * (1 + np.sqrt(self.n)) / (1 + node.n)
                if score > best_score:
                    best_score = score
                    which = action
            if which is None:
                self.n = 1
                self.deterministic = True
            else:
                self.n += 1
                state.move(*which)
                node = self.p[which][1]
                node.grow(agent, state, depth + 1)
                state.rewind()
            children = list(map(lambda t: t[1], self.p.values()))
            node = max(filter(lambda n: n.n > 0, children), key=lambda n: -n.v)
            if -node.v == INFINITY and node.deterministic:
                self.n = 1
                self.deterministic = True
            elif reduce(lambda a, b: a and b, map(lambda n: n.n > 0, children)) and -node.v == -INFINITY and node.deterministic:
                self.n = 1
                self.deterministic = True
            self.w = -node.v
