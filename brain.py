from evaluate import evaluate

NEIGHBORS  = [( i,  0) for i in range(1, 5)]
NEIGHBORS += [(-i,  0) for i in range(1, 5)]
NEIGHBORS += [( 0,  i) for i in range(1, 5)]
NEIGHBORS += [( 0, -i) for i in range(1, 5)]
NEIGHBORS += [( i,  i) for i in range(1, 5)]
NEIGHBORS += [( i, -i) for i in range(1, 5)]
NEIGHBORS += [(-i,  i) for i in range(1, 5)]
NEIGHBORS += [(-i, -i) for i in range(1, 5)]

class Agent():
    def getAction(self, state):
        assert False, "not implemented"


class ReflexAgent(Agent):
    def getAction(self, state):
        return max([a for a in self.generateActions(state)],
            key=lambda a: evaluate(state.update(a[0], a[1], state.AI)))
    
    def generateActions(self, state):
        actions = set()
        for x, y, _ in state.hist:
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if state.inBound(nx, ny) and state.isEmpty(nx, ny):
                    actions.add((nx, ny))
        return actions


class GameData():
    GRID_SIZE = 15
    EMPTY     = 0
    AI        = 1
    HUMAN     = 2

    def __init__(self, first, prev=None, hist=[], agent=ReflexAgent()):
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

    def update(self, x, y, who):
        copy = [col[:] for col in self.moves]
        copy[x][y] = who
        hist = self.hist + [ (x, y, who) ]
        return GameData(self.first, copy, hist, self.agent)
    
    def think(self):
        return self.agent.getAction(self)

    def other(self, who):
        return self.AI + self.HUMAN - who
