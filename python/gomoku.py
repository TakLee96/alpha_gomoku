""" Gomoku Game State """

import numpy as np

N = 15  # board size
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

class State:
  def __init__(self):
    self.player = 1  # 1 for black and -1 for white
    self.board = np.zeros(shape=(N, N), dtype=np.int8)
    self.history = list()
    self.end = False

  def _count(self, x, y, dx, dy):
    if x >= 15 or x < 0 or y >= 15 or y < 0 or self.board[x, y] != self.player:
      return 0
    return 1 + self._count(x + dx, y + dy, dx, dy)

  def _win(self, x, y):
    for dx, dy in DIRECTIONS:
      if self._count(x + dx, y + dy, dx, dy) + \
        self._count(x - dx, y - dy, -dx, -dy) >= 4:
        return True
    return False

  def move(self, x, y):
    assert self.board[x, y] == 0 and not self.end
    self.board[x, y] = self.player
    self.history.append((x, y))
    if self._win(x, y):
      self.end = True
    else:
      self.player = -self.player
    return self.end

  def rewind(self):
    x, y = history.pop()
    self.board[x, y] = 0
    if self.end:
      self.end = False
    else:
      self.player = -self.player

  def __str__(self):
    return str(self.board).replace("-1", "x").replace(" 1", "o").replace(" 0", "+")
