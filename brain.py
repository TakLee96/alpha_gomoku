from evaluate import evaluate, normalize, length


NEIGHBORS  = [( i,  0) for i in range(1, 5)]
NEIGHBORS += [(-i,  0) for i in range(1, 5)]
NEIGHBORS += [( 0,  i) for i in range(1, 5)]
NEIGHBORS += [( 0, -i) for i in range(1, 5)]
NEIGHBORS += [( i,  i) for i in range(1, 5)]
NEIGHBORS += [( i, -i) for i in range(1, 5)]
NEIGHBORS += [(-i,  i) for i in range(1, 5)]
NEIGHBORS += [(-i, -i) for i in range(1, 5)]

INFINITY = 1000000.0
DEPTH = 2


class Agent():
    def value(self, state):
        return evaluate(state)

    def getAction(self, state):
        assert False, "not implemented"


class ReflexAgent(Agent):
    def getAction(self, state):
        return max([a for a in self.generateActions(state)],
            key=lambda a: self.value(state.update(a[0], a[1], state.AI)))
    
    def generateActions(self, state):
        actions = set()
        for x, y, _ in state.hist:
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if state.inBound(nx, ny) and state.isEmpty(nx, ny):
                    actions.add((nx, ny))
        return actions


class AlphaBetaAgent(ReflexAgent):
    def value(self, state, depth, who, alpha, beta):
        if who == state.AI:
            depth += 1
        if depth == DEPTH:
            return evaluate(state)
        if state.isWin(who):
            return INFINITY
        if state.isLose(who):
            return -INFINITY
        if who == state.AI:
            return self.max_value(state, depth, who, alpha, beta)[0]
        return self.min_value(state, depth, who, alpha, beta)

    def max_value(self, state, depth, who, alpha, beta):
        v, chosen = -INFINITY, None
        for action in self.generateActions(state):
            val = self.value(state.update(action[0], action[1], who), depth, state.other(who), alpha, beta)
            if val > v:
                v, chosen = val, action
            if v > beta:
                return (v, chosen)
            alpha = max(alpha, v)
        return (v, chosen)

    def min_value(self, state, depth, who, alpha, beta):
        v = INFINITY
        for action in self.generateActions(state):
            val = self.value(state.update(action[0], action[1], who), depth, state.other(who), alpha, beta)
            if val < v:
                v = val
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def getAction(self, state):
        val, action = self.max_value(state, 0, state.AI, -INFINITY, INFINITY)
        assert action is not None
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
        if first == 1:
            self.moves[self.GRID_SIZE/2][self.GRID_SIZE/2] = self.AI
            self.hist.append((self.GRID_SIZE/2, self.GRID_SIZE/2, first))

    def inBound(self, x, y):
        return 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE

    def isEmpty(self, x, y):
        return self.moves[x][y] == self.EMPTY

    def isWin(self, who):
        x, y, w = self.hist[len(self.hist)-1]
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

    def update(self, x, y, who):
        assert self.isEmpty(x, y)
        copy = [col[:] for col in self.moves]
        copy[x][y] = who
        hist = self.hist + [ (x, y, who) ]
        return GameData(self.first, copy, hist, self.agent)
    
    def think(self):
        return self.agent.getAction(self)

    def other(self, who):
        return self.AI + self.HUMAN - who
