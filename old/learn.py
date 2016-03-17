from agent import RandomAgent, ReflexAgent, AlphaBetaAgent
from random import choice, random
from game import GameState

class Counter(dict):

    def __init__(self):
        self.max = 1.0
        dict.__init__(self)

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        return 0.0

    def __setitem__(self, key, value):
        self.max = max(self.max, abs(value))
        dict.__setitem__(self, key, value)

    def normalize(self):
        for key in self.keys():
            dict.__setitem__(self, key, dict.__getitem__(self, key) / self.max)
        self.max = 1.0

class ApproximateQLearningAgent():

    def __init__(self, alpha=0.1, epsilon=0.2, gamma=0.99, numTraining=100, numTesting=20,
                 debug=False, index=GameState.HUMAN, opponent=RandomAgent()):
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)
        self.numTesting = int(numTesting)
        self.weights = Counter()
        self.index = index
        self.debug = debug
        self.opponent = opponent

    def getQValue(self, state, action):
        state.move(action)
        features = self.extractFeatures(state)
        total = sum([self.weights[f] * features[f] for f in features.keys()])
        state.rewind()
        return total

    def getValue(self, state):
        legalActions = state.getLegalActions()
        if len(legalActions) == 0:
            return 0.0
        return max([self.getQValue(state, action) for action in legalActions])

    def getPolicy(self, state):
        legalActions = state.getLegalActions()
        if len(legalActions) == 0:
            return None
        return max(legalActions, key=lambda a: self.getQValue(state, a))

    def getAction(self, state):
        legalActions = state.getLegalActions()
        if len(legalActions) == 0:
            return None
        if random() < self.epsilon:
            return choice(legalActions)
        else:
            return self.getPolicy(state)

    def update(self, s):
        action = self.getAction(s)
        oppaction = None
        reward = 1.0
        msg = "going"
        if action == None:
            return "tie"
        s.move(action)
        if s.isWin(self.index):
            reward = 100.0
            msg = "win"
        else:
            oppaction = self.opponent.getAction(s)
            if oppaction == None:
                return "tie"
            s.move(oppaction)
            if s.isLose(self.index):
                reward = -100.0
                msg = "lose"
        # do the necessary update here
        diff = reward + self.discount * self.getValue(s)
        s.rewind()
        if oppaction != None:
            s.rewind()
        diff = diff - self.getQValue(s, action)
        features = self.extractFeatures(s)
        for feature in features:
            self.weights[feature] = self.weights[feature] + self.alpha * diff * features[feature]
        self.weights.normalize()
        s.move(action)
        if oppaction is not None:
            s.move(oppaction)
        if self.debug:
            print action, oppaction
            print s
            print self.weights
        return msg

    def __check(self, state, move, direction, checked, features):
        x, y = move
        dx, dy = direction
        who = state.moves[x][y]
        if who == self.index:
            one = "o"
            other = "x"
        else:
            one = "x"
            other = "o"
        feature = one
        nx, ny = x + dx, y + dy
        jumped = False
        while state.inBound(nx, ny):
            next = state.moves[nx][ny]
            if next == who:
                feature = feature + one
                checked.add(((nx, ny), direction))
                nx, ny = nx + dx, ny + dy
            elif next == state.EMPTY and state.inBound(nx, ny) and state.moves[nx][ny] == who and not jumped:
                jumped = True
                feature = feature + "-"
                nx, ny = nx + dx, ny + dy
            else:
                if next == state.EMPTY:
                    feature = feature + "-"
                else:
                    feature = feature + other
                break
        if not state.inBound(nx, ny):
            feature = feature + other
        nx, ny = x - dx, y - dy
        jumped = False
        while state.inBound(nx, ny):
            next = state.moves[nx][ny]
            if next == who:
                feature = one + feature
                checked.add(((nx, ny), direction))
                nx, ny = nx - dx, ny - dy
            elif next == state.EMPTY and state.inBound(nx, ny) and state.moves[nx][ny] == who and not jumped:
                jumped = True
                feature = "-" + feature
                nx, ny = nx - dx, ny - dy
            else:
                if next == state.EMPTY:
                    feature = "-" + feature
                else:
                    feature = other + feature
                break
        if not state.inBound(nx, ny):
            feature = other + feature
        features[feature] = features[feature] + 1

    def extractFeatures(self, state):
        checked = set()
        features = Counter()
        for i, move in enumerate(state.hist):
            if move != None:
                for direction in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
                    if (move, direction) not in checked:
                        checked.add((move, direction))
                        self.__check(state, move, direction, checked, features)
        return features

    def doneUpdating(self):
        self.alpha = 0.0
        self.epsilon = 0.0

    def beginTraining(self, state):
        count = 0
        for i in range(self.numTraining):
            s = state.copy()
            msg = "going"
            while msg == "going":
                msg = self.update(s)
            if msg == "win":
                count += 1
            print i+1, "games played", msg, "winning rate", 1.0*count/(i+1)
        print self.weights
        self.doneUpdating()
        count = 0
        for j in range(self.numTesting):
            s = state.copy()
            msg = "going"
            while msg == "going":
                msg = self.update(s)
            if msg == "win":
                count += 1
            print "test", j+1, msg
        print "during testing, win", count, "out of", self.numTesting, "games"



if __name__ == '__main__':
    agent = ApproximateQLearningAgent(numTraining=100, numTesting=20, debug=False)
    state = GameState(agent.index)
    agent.beginTraining(state)
