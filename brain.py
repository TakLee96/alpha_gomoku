from evaluate import evaluate, normalize, length

INFINITY = 10000000000.0
DEPTH = 2

class Agent():
    def value(self, state):
        return evaluate(state)

    def getAction(self, state):
        assert False, "not implemented"


class ReflexAgent(Agent):
    def getAction(self, state):
        return max([a for a in self.generateActions(state)],
            key=lambda a: self.value(state.generateSuccessor(a[0], a[1], state.AI)))
    
    def generateActions(self, state):
        actions = set()
        for x, y in state.hist:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, 1)]:
                nx, ny = x + dx, y + dy
                if state.inBound(nx, ny) and state.isEmpty(nx, ny):
                    actions.add((nx, ny))
        return actions


class AlphaBetaAgent(ReflexAgent):
    def value(self, state, depth, alpha, beta):
        who = state.next()
        if who == state.first:
            depth += 1
        if state.isWin(state.AI):
            return INFINITY - depth / 4
        if state.isLose(state.AI):
            return -INFINITY + depth / 4
        if depth == DEPTH:
            return evaluate(state)
        if who == state.AI:
            return self.max_value(state, depth, alpha, beta)[0]
        return self.min_value(state, depth, alpha, beta)

    def max_value(self, state, depth, alpha, beta):
        v, chosen = -INFINITY, None
        for action in self.generateActions(state):
            val = self.value(state.generateSuccessor(action[0], action[1]), depth, alpha, beta)
            if val > v:
                v, chosen = val, action
            if v > beta:
                return (v, chosen)
            alpha = max(alpha, v)
        return (v, chosen)

    def min_value(self, state, depth, alpha, beta):
        v = INFINITY
        for action in self.generateActions(state):
            val = self.value(state.generateSuccessor(action[0], action[1]), depth, alpha, beta)
            if val < v:
                v = val
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def getAction(self, state):
        val, action = self.max_value(state, 0, -INFINITY, INFINITY)
        return action


class GameData():
    GRID_SIZE = 15
    EMPTY     = 0
    AI        = 1
    HUMAN     = 2

    def __init__(self, first, prev=None, hist=[], agent=AlphaBetaAgent()):
        self.moves = prev or [[self.EMPTY for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.first = first
        self.hist  = hist
        self.agent = agent

    def inBound(self, x, y):
        return 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE

    def isEmpty(self, x, y):
        return self.moves[x][y] == self.EMPTY

    def isWin(self, who):
        x, y = self.hist[len(self.hist)-1]
        w = self.moves[x][y]
        if w != who:
            return False
        _, _, a = length(normalize(x, y, 1, 0, -1, 0, who, self), self)
        if a > 4:
            return True
        _, _, b = length(normalize(x, y, 0, 1, 0, -1, who, self), self)
        if b > 4:
            return True
        _, _, c = length(normalize(x, y, 1, 1, -1, -1, who, self), self)
        if c > 4:
            return True
        _, _, d = length(normalize(x, y, 1, -1, -1, 1, who, self), self)
        if d > 4:
            return True
        return False

    def isLose(self, who):
        return self.isWin(self.other(who))

    def next(self):
        if len(self.hist) % 2 == 0:
            return self.first
        return self.other(self.first)

    def generateSuccessor(self, x, y):
        who = self.next()
        copy = [col[:] for col in self.moves]
        copy[x][y] = who
        hist = self.hist + [ (x, y) ]
        return GameData(self.first, copy, hist, self.agent)
    
    def think(self):
        return self.agent.getAction(self)

    def other(self, who):
        return self.AI + self.HUMAN - who
