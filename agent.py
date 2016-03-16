from evaluate import evaluate, normalize, length
from random import choice
from game import GameState

INFINITY = 10000000000.0
DEPTH = 3

# TODO: The AI is still very slow, why?
# TODO: It seems that the AI is still making silly mistakes

class Agent():
    def __init__(self, index=GameState.AI):
        self.index = index

    def value(self, state):
        return evaluate(state)

    def getAction(self, state):
        assert False, "not implemented"


class RandomAgent():
    def getAction(self, state):
        return choice(state.getLegalActions())


class ReflexAgent(Agent):
    def getAction(self, state):
        def key(action):
            state.move(action)
            val = self.value(state)
            state.rewind()
            return val
        return max([a for a in state.getLegalActions()], key=key)


class AlphaBetaAgent(ReflexAgent):
    def value(self, state, depth, alpha, beta):
        who = state.next()
        depth += 1
        if state.isLose(self.index):
            return -INFINITY + depth / 4.0
        if state.isWin(self.index):
            return INFINITY - depth / 4.0
        if depth == DEPTH:
            # TODO: our current evaluation function is VERY expensive
            # Can we avoid looking redundantly at every hist move?
            return evaluate(state)
        if who == self.index:
            return self.max_value(state, depth, alpha, beta)[0]
        return self.min_value(state, depth, alpha, beta)

    def max_value(self, state, depth, alpha, beta):
        v, chosen = -INFINITY, None
        for action in state.getLegalActions():
            state.move(action)
            val = self.value(state, depth, alpha, beta)
            state.rewind()
            if val > v:
                v, chosen = val, action
            if v > beta:
                return (v, chosen)
            alpha = max(alpha, v)
        return (v, chosen)

    def min_value(self, state, depth, alpha, beta):
        v = INFINITY
        for action in state.getLegalActions():
            state.move(action)
            val = self.value(state, depth, alpha, beta)
            state.rewind()
            if val < v:
                v = val
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def getAction(self, state):
        val, action = self.max_value(state, 0, -INFINITY, INFINITY)
        return action