from evaluate import evaluate, normalize, length

INFINITY = 10000000000.0
DEPTH = 3

# TODO: The AI is still very slow, why?
# TODO: It seems that the AI is still making silly mistakes

class Agent():
    def value(self, state):
        return evaluate(state)

    def getAction(self, state):
        assert False, "not implemented"


class ReflexAgent(Agent):
    def getAction(self, state):
        def key(action):
            state.move(action[0], action[1])
            val = self.value(state)
            state.rewind()
            return val
        return max([a for a in self.generateActions(state)], key=key)
    
    def generateActions(self, state):
        # TODO: I should revise the generateActions function to do some computation
        # and propose less but better moves to consider to reduce branch factor
        actions = set()
        for x, y in state.hist:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                # TODO: this is considering less points
                nx, ny = x + dx, y + dy
                if state.inBound(nx, ny) and state.isEmpty(nx, ny):
                    actions.add((nx, ny))
        return actions


class AlphaBetaAgent(ReflexAgent):
    def value(self, state, depth, alpha, beta):
        who = state.next()
        depth += 1
        if state.isWin(state.AI):
            return INFINITY - depth / 4.0
        if state.isLose(state.AI):
            return -INFINITY + depth / 4.0
        if depth == DEPTH:
            # TODO: our current evaluation function is VERY expensive
            # Can we avoid looking redundantly at every hist move?
            return evaluate(state)
        if who == state.AI:
            return self.max_value(state, depth, alpha, beta)[0]
        return self.min_value(state, depth, alpha, beta)

    def max_value(self, state, depth, alpha, beta):
        v, chosen = -INFINITY, None
        for action in self.generateActions(state):
            state.move(action[0], action[1])
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
        for action in self.generateActions(state):
            state.move(action[0], action[1])
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

AGENTS = {'m': AlphaBetaAgent, 'r': ReflexAgent}

class GameData():
    GRID_SIZE = 15
    EMPTY     = 0
    AI        = 1
    HUMAN     = 2

    def __init__(self, first, agent=AlphaBetaAgent()):
        self.moves = [[self.EMPTY for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.first = first
        self.hist  = []
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
        _, _, a = length(normalize(x, y, 1, 0, who, self), self)
        if a > 4:
            return True
        _, _, b = length(normalize(x, y, 0, 1, who, self), self)
        if b > 4:
            return True
        _, _, c = length(normalize(x, y, 1, 1, who, self), self)
        if c > 4:
            return True
        _, _, d = length(normalize(x, y, 1, -1, who, self), self)
        if d > 4:
            return True
        return False

    def isLose(self, who):
        return self.isWin(self.other(who))

    def next(self):
        if len(self.hist) % 2 == 0:
            return self.first
        return self.other(self.first)

    def move(self, x, y):
        who = self.next()
        self.moves[x][y] = who
        self.hist.append( (x, y) )

    def rewind(self):
        x, y = self.hist.pop()
        self.moves[x][y] = self.EMPTY
    
    def think(self):
        return self.agent.getAction(self)

    def other(self, who):
        return self.AI + self.HUMAN - who
