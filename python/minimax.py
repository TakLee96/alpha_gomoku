""" alpha-beta minimax agent """
from feature import diff
from scipy.signal import convolve2d as conv2d


INFINITY = 1e10
FILTER = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [1, 0, 1, 0, 1],
]).astype(np.int8)


class MinimaxAgent:
    def __init__(self, depth=5)
        self.depth = depth

    def _policy(self, state):
        adjacent = conv2d(np.abs(state.board), FILTER, mode="same")
        if state.player == 1:
            four_danger = state.features["four-x"] + state.features["-xxxxo"]
            three_danger = state.features["-x-xx-"] + state.features["-xxx-"]
        else:
            four_danger = state.features["four-o"] + state.features["-oooox"]
            three_danger = state.features["-o-oo-"] + state.features["-ooo-"]
        actions = []
        for x in range(15):
            for y in range(15):
                if adjacent[x, y] > 0 and state.board[x, y] == 0:
                    new, old = diff(state, x, y)
                    if (not state._long(j, k) and new["-o-oo-"] + new["-ooo-"] < 2 and
                        new["four-o"] + new["-oooo-"] + new["-oooox"] < 2):
                        if new["win-o"] > 0 or new["win-x"] > 0:
                            return [(x, y)]
                        elif state.player == 1:
                            if four_danger > 0:
                                if four_danger + new["four-x"] + new["-xxxxo"] == 0:
                                    actions.append((x, y))
                            elif three_danger > 0:
                                if new["four-x"] > 0 or new["-xxxxo"] > 0
                        else:


        prob = (adjacent > 0).reshape(225).astype(float)
        prob = prob / prob.sum()
        random_action = np.random.choice(225, p=prob)



    def _max_value(self, state, alpha, beta, depth):
        if state.end:
            return 0.0
        if depth == self.depth:
            return self.value(state.player * state.board)
        max_value = 0.0
        for x, y in self.policy(state.player * state.board):
            state.move(x, y)
            max_value = max(max_value, self._min_value(state, alpha, beta, depth + 1))
            if max_value > beta:
                return max_value
            alpha = max(alpha, max_value)
            state.rewind()
        return max_value

    def _min_value(self, state, alpha, beta, depth):
        if state.end:
            return 1.0
        if depth == self.depth:
            return 1.0 - self.value(state.player * state.board)
        min_value = 1.0
        for x, y in self.policy(state.player * state.board):
            state.move(x, y)
            min_value = min(min_value, self._max_value(state, alpha, beta, depth + 1))
            if min_value < alpha:
                return min_value
            beta = min(beta, min_value)
            state.rewind()
        return min_value

    def get_action(self, state):
        actions = self.policy(state.board)
        def key_func(action):
            x, y = action
            state.move(x, y)
            val = self._min_value(state, 0.0, 1.0, 1)
            state.rewind()
            return val
        return max(actions, key=key_func)
