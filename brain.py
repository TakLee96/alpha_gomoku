from random import random

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
            key=lambda a: self.value(state.update(a[0], a[1], AI)))
    
    def generateActions(self, state):
        actions = set()
        for x, y, _ in state.hist:
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if state.inBound(nx, ny) and state.isEmpty(nx, ny):
                    actions.add((nx, ny))
        return actions
    
    def normalize(self, x, y, ax, ay, bx, by, who, state):
        seq = [0 for _ in range(9)]
        seq[4] = who
        for i in range(1, 5):
            nx, ny = x + i * ax, y + i * ay
            if state.inBound(nx, ny):
                seq[4-i] = state.moves[nx][ny]
            else:
                seq[4-i] = other(who)
        for i in range(1, 5):
            nx, ny = x + i * bx, y + i * by
            if state.inBound(nx, ny):
                seq[4+i] = state.moves[nx][ny]
            else:
                seq[4+i] = other(who)
        return seq

    def length(self, seq):
        who = seq[4]
        copy = seq[:]
        left = 4
        right = 4
        for i in range(5, 9):
            if seq[i] != who:
                break
            right = i
        for i in range(3, -1, -1):
            if seq[i] != who:
                break
            left = i
        return (left, right, right - left + 1)

    def countEmpty(self, seq):
        who = seq[4]
        leftCount = 0
        rightCount = 0
        for i in range(5, 9):
            if seq[i] == EMPTY:
                leftCount += 1
            elif seq[i] == other(who):
                break
        for i in range(3, -1, -1):
            if seq[i] == EMPTY:
                rightCount += 1
            elif seq[i] == other(who):
                break
        return (leftCount, rightCount)

    def checkfour(self, seq, left, right):
        leftCount, rightCount = self.countEmpty(seq)
        if leftCount != 0 and rightCount != 0:
            return DEATH
        elif leftCount != 0 or rightCount != 0:
            return URGENT
        return USELESS

    def checkthree(self, seq, left, right):
        leftCount, rightCount = self.countEmpty(seq)
        who = seq[4]
        if (seq[left-1] == EMPTY and seq[left-2] == who):
            return URGENT
        if (seq[right+1] == EMPTY and seq[right+2] == who):
            return URGENT
        if leftCount != 0 and rightCount != 0:
            if leftCount + rightCount > 2:
                return URGENT
            else:
                return IMPORTANT
        elif leftCount != 0 or rightCount != 0:
            if leftCount + rightCount > 2:
                return IMPORTANT
        return USELESS

    def checktwo(self, seq, left, right):
        leftCount, rightCount = self.countEmpty(seq)
        who = seq[4]
        # TODO: will 2 jump 2 or 1 jump 2 jump 1 ever occur
        if (seq[left-1] == EMPTY and seq[left-2] == who):
            return self.checkthree([ other(who) ] + seq[:left-1] + seq[left:], left-1, right)
        if (seq[right+1] == EMPTY and seq[right+2] == who):
            return self.checkthree(seq[:right+1] + seq[right+2:] + [ other(who) ], left, right+1)
        if leftCount == 0 and rightCount == 0:
            return USELESS
        if leftCount == 0 or rightCount == 0:
            return OKAY
        if leftCount + rightCount < 3:
            return USELESS
        return GOOD

    def score(self, seq):
        who = seq[4]
        left, right, length = self.length(seq)
        if length >= 5:
            return DEATH
        elif length == 4:
            return self.checkfour(seq, left, right)
        elif length == 3:
            return self.checkthree(seq, left, right)
        elif length == 2:
            return self.checktwo(seq, left, right)
        elif length == 1:
            return USELESS # TODO: can improve by looking at jumping 2

    def value(self, state):
        score = 0
        for x, y, who in state.hist:
            temp  = self.score(self.normalize(x, y, 1, 0, -1, 0, who, state))
            temp += self.score(self.normalize(x, y, 0, 1, 0, -1, who, state))
            temp += self.score(self.normalize(x, y, 1, 1, -1, -1, who, state))
            temp += self.score(self.normalize(x, y, 1, -1, -1, 1, who, state))
            multiplier = 1.0 if who == AI else -20.1
            score += multiplier * temp
        return score + random()


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
