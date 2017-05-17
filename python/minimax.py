""" Minimax Agent with Alpha-Beta pruning """

class Agent():
  def __init__(self, player, policy):
    self.player = player
    self.policy = policy

  def get_action(self, state):
    assert self.player == state.player
    actions = self.policy(state.player * state.board)
    return actions[0]


class MinimaxAgent:
  def __init__(self, player, policy, value, depth=5)
    self.player = player
    self.policy = policy  # a function
    self.value = value    # a function
    self.depth = depth

  def _max_value(self, state, alpha, beta, depth):
    assert self.player == state.player
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
    assert self.player != state.player
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
    assert self.player == state.player
    actions = self.policy(state.player * state.board)
    def key_func(action):
      x, y = action
      state.move(x, y)
      val = self._min_value(state, 0.0, 1.0, 1)
      state.rewind()
      return val
    return max(actions, key=key_func)
