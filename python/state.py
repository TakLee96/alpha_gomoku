""" gomoku game state representation """
import numpy as np
from feature import diff, defaultdict


DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]


class State:
    def __init__(self):
        self.player = 1    # 1 for black and -1 for white
        self.board = np.zeros(shape=(15, 15), dtype=np.int8)
        self.history = list()
        self.features = defaultdict()
        self.end = False
        self.violate = False

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

    def move(self, x, y):
        assert self.board[x, y] == 0 and not self.end
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
