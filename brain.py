from random import random
from evaluate import evaluate

GRID_SIZE = 15

DEATH     = 10000.0
URGENT    = 1000.0
IMPORTANT = 100.0
GOOD      = 10.0
OKAY      = 1.0
USELESS   = 0.0

EMPTY = 0
AI    = 1
HUMAN = 2
other = lambda w: AI + HUMAN - w

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
            key=lambda a: evaluate(state.update(a[0], a[1], AI)))
    
    def generateActions(self, state):
        actions = set()
        for x, y, _ in state.hist:
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if state.inBound(nx, ny) and state.isEmpty(nx, ny):
                    actions.add((nx, ny))
        return actions


class GameData():
    def __init__(self, first, prev=None, hist=[], agent=ReflexAgent()):
        self.moves = prev or [[EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.first = first
        self.hist  = hist
        self.agent = agent
        if first == 1:
            self.moves[GRID_SIZE/2][GRID_SIZE/2] = AI
            self.hist.append((GRID_SIZE/2, GRID_SIZE/2, first))

    def inBound(self, x, y):
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def isEmpty(self, x, y):
        return self.moves[x][y] == EMPTY

    def update(self, x, y, who):
        copy = [col[:] for col in self.moves]
        copy[x][y] = who
        hist = self.hist + [ (x, y, who) ]
        return GameData(self.first, copy, hist, self.agent)
    
    def think(self):
        return self.agent.getAction(self)
