class GameState():
    GRID_SIZE = 15
    EMPTY     = 0
    AI        = 1
    HUMAN     = 2

    def __init__(self, first, hist=[], prev=None):
        self.moves = prev or [[self.EMPTY for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.first = first
        self.hist  = hist

    def inBound(self, x, y):
        return 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE

    def isEmpty(self, x, y):
        return self.moves[x][y] == self.EMPTY

    def __count(self, x, y, dx, dy, w):
        x, y = x + dx, y + dy
        if not self.inBound(x, y) or self.moves[x][y] != w:
            return 0
        return 1 + self.__count(x, y, dx, dy, w)

    def isWin(self, who):
        if len(self.hist) == 0:
            return False
        move = self.hist[len(self.hist)-1]
        if move == None:
            return False
        x, y = move
        w = self.moves[x][y]
        if w != who:
            return False
        if (self.__count(x, y, 1, 0, w)  + 1 + self.__count(x, y, -1, 0, w)  >= 5 or
            self.__count(x, y, 0, 1, w)  + 1 + self.__count(x, y, 0, -1, w)  >= 5 or
            self.__count(x, y, 1, 1, w)  + 1 + self.__count(x, y, -1, -1, w) >= 5 or
            self.__count(x, y, 1, -1, w) + 1 + self.__count(x, y, -1, 1, w)  >= 5):
           return True
        return False

    def isLose(self, who):
        return self.isWin(self.other(who))

    def next(self):
        if len(self.hist) % 2 == 0:
            return self.first
        return self.other(self.first)

    def move(self, action):
        assert action is not None
        x, y = action
        who = self.next()
        self.moves[x][y] = who
        self.hist.append(action)

    def rewind(self):
        action = self.hist.pop()
        if action == None:
            self.terminated = False
        else:
            x, y = action
            self.moves[x][y] = self.EMPTY

    def other(self, who):
        return self.AI + self.HUMAN - who

    def getLegalActions(self):
        # TODO: I should revise the generateActions function to do some computation
        # and propose less but better moves to consider to reduce branch factor
        if len(self.hist) == 0:
            return [(self.GRID_SIZE/2, self.GRID_SIZE/2)]
        actions = set()
        for move in self.hist:
            if move != None:
                x, y = move
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
                               (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (2, -2), (-2, 2), (-2, -2)]:
                    nx, ny = x + dx, y + dy
                    if self.inBound(nx, ny) and self.isEmpty(nx, ny):
                        actions.add((nx, ny))
        return list(actions)

    def copy(self):
        return GameState(self.first, self.hist[:], [col[:] for col in self.moves])

    def __str__(self):
        string = "".join(["-" for _ in range(self.GRID_SIZE+2)]) + "\n"
        convert = { 0: "+", 1: "o", 2: "x" }
        for i in range(self.GRID_SIZE):
            string += "|"
            for j in range(self.GRID_SIZE):
                string += convert[self.moves[i][j]]
            string += "|\n"
        string += "".join(["-" for _ in range(self.GRID_SIZE+2)])
        return string
