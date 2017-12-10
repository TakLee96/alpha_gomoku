""" gomoku game state representation """

import pickle
import numpy as np
from feature import diff, defaultdict
from scipy.signal import convolve2d as conv2d

DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]
FILTER = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [1, 0, 1, 0, 1],
]).astype(np.int8)

class State:
    def __init__(self):
        self.player = 1    # 1 for black and -1 for white
        self.board = np.zeros(shape=(15, 15), dtype=np.int8)
        self.history = list()
        self.features = defaultdict()
        self.end = False
        self.violate = False

    def copy(self):
        cloned = State()
        cloned.player = self.player
        cloned.board = self.board.copy()
        cloned.history = list(self.history)
        cloned.features = defaultdict(self.features)
        cloned.end = self.end
        cloned.violate = self.violate
        return cloned

    def _build(self, x, y, dx, dy, result):
        if 0 <= x <= 14 and 0 <= y <= 14 and self.board[x, y] == self.player:
            result.append((x, y))
            self._build(x + dx, y + dy, dx, dy, result)

    def highlight(self, x, y):
        assert self.end
        for dx, dy in DIRECTIONS:
            result = [(x, y)]
            self._build(x + dx, y + dy,  dx,  dy, result)
            self._build(x - dx, y - dy, -dx, -dy, result)
            if len(result) >= 5:
                return result
        raise Exception("wrong call to highlight")

    def feature_move(self, x, y):
        assert self.board[x, y] == 0
        new, old = diff(self, x, y)
        self.features.add(new)
        self.features.sub(old)
        self.board[x, y] = self.player
        self.player = -self.player

    def move(self, x, y):
        assert self.board[x, y] == 0 and not self.end
        assert x >= 0 and y >= 0, "negative input, might mean resign"
        new, old = diff(self, x, y)
        self.features.add(new)
        self.features.sub(old)
        self.board[x, y] = self.player
        self.history.append((x, y))
        if "win-o" in new or "win-x" in new:
            self.end = True
        else:
            if self.player == 1 and (new["-o-oo-"] + new["-ooo-"] >= 2 or
                new["four-o"] + new["-oooo-"] + new["-oooox"] >= 2 or "violate" in new):
                self.end = True
                self.violate = True
            self.player = -self.player
        return self.end

    def rewind(self):
        x, y = self.history.pop()
        self.board[x, y] = 0
        self.player = 1 if len(self.history) % 2 == 0 else -1
        new, old = diff(self, x, y)
        self.features.sub(new)
        self.features.add(old)
        self.end = False
        self.violate = False

    def __str__(self):
        return str(self.board).replace("-1", "x").replace(" 1", "o").replace(" 0", "+")

    def adjacent(self):
        return conv2d(np.abs(self.board), FILTER, mode="same")

    def featurize(self):
        features = np.zeros((15, 15, 11), dtype=np.float32)
        features[:, :, 0] = self.board > 0   # black
        features[:, :, 1] = self.board < 0   # white
        features[:, :, 2] = self.board == 0  # empty
        features[0, :, 3] = 1
        features[14,:, 4] = 1
        features[:, 0, 5] = 1
        features[:,14, 6] = 1
        if self.player == 1:
            features[:, :, 7] = 1
        else:
            features[:, :, 8] = 1
        adjacent = self.adjacent()
        for x in range(15):
            for y in range(15):
                if adjacent[x, y] > 0:
                    if self.player == 1:
                        new, old = diff(self, x, y)
                        if (new["-o-oo-"] + new["-ooo-"] >= 2 or "violate" in new or
                            new["four-o"] + new["-oooo-"] + new["-oooox"] >= 2):
                            features[:, :, 9] = 1
                    else:
                        self.player = 1
                        new, old = diff(self, x, y)
                        if (new["-o-oo-"] + new["-ooo-"] >= 2 or "violate" in new or
                            new["four-o"] + new["-oooo-"] + new["-oooox"] >= 2):
                            features[:, :, 10] = 1
                        self.player = -1
        return features

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({"moves": self.history, "violate": self.violate}, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
