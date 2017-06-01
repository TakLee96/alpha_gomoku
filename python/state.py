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

    def _count(self, x, y, dx, dy):
        if x >= 15 or x < 0 or y >= 15 or y < 0 or self.board[x, y] != self.player:
            return 0
        return 1 + self._count(x + dx, y + dy, dx, dy)

    def _win(self, x, y):
        for dx, dy in DIRECTIONS:
            count = self._count(x + dx, y + dy, dx, dy) + \
                self._count(x - dx, y - dy, -dx, -dy)
            if count == 4 or (count > 4 and self.player == -1):
                return True
        return False

    def _long(self, x, y):
        for dx, dy in DIRECTIONS:
            if self._count(x + dx, y + dy, dx, dy) + \
                self._count(x - dx, y - dy, -dx, -dy) > 4:
                return True
        return False

    def _build(self, x, y, dx, dy, result):
        if 0 <= x <= 14 and 0 <= y <= 14 and self.board[x, y] == self.player:
            result.append((x, y))
            self._build(x + dx, y + dy, dx, dy, result)

    def highlight(self, x, y):
        assert self.end
        for dx, dy in DIRECTIONS:
            if self._count(x + dx, y + dy, dx, dy) + \
                self._count(x - dx, y - dy, -dx, -dy) >= 4:
                break
        result = [(x, y)]
        self._build(x + dx, y + dy,  dx,  dy, result)
        self._build(x - dx, y - dy, -dx, -dy, result)
        return result

    def move(self, x, y):
        assert self.board[x, y] == 0 and not self.end
        new = diff(self, x, y)
        self.features.add(new)
        self.board[x, y] = self.player
        self.history.append((x, y))
        if self._win(x, y):
            self.end = True
        else:
            if new["-o-oo-"] + new["-ooo-"] >= 2 or \
                new["four-o"] + new["-oooo-"] >= 2 or self._long(x, y):
                self.end = True
            self.player = -self.player
        return self.end

    def rewind(self):
        x, y = self.history.pop()
        self.board[x, y] = 0
        self.features.sub(diff(self, x, y))
        if self.end:
            self.end = False
        self.player = 1 if len(self.history) % 2 == 0 else -1

    def __str__(self):
        return str(self.board).replace("-1", "x").replace(" 1", "o").replace(" 0", "+")
