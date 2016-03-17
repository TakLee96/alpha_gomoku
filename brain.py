from game import GameState
from random import choice

class Counter(dict):
    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        return 0

def __check(state, move, direction, checked, features):
    x, y = move
    dx, dy = direction
    who = state.moves[x][y]
    if who == state.first:
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
    if len(feature) >= 7:
        if feature[1] == "o":
            features["*ooooo*"] = features["*ooooo*"] + 1
        else:
            features["*xxxxx*"] = features["*xxxxx*"] + 1
    elif len(feature) > 3 and (feature[0] == "-" or feature[-1] == "-"):
        if feature[0] == "-" and feature[-1] != "-":
            feature = feature[::-1]
        features[feature] = features[feature] + 1

def extractFeatures(state):
    checked = set()
    features = Counter()
    for move in state.hist:
        if move != None:
            for direction in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
                if (move, direction) not in checked:
                    checked.add((move, direction))
                    __check(state, move, direction, checked, features)
    return features

weight = {
    "-oo-":    1e3,
    "-xx-":   -1e2,
    "xoo-":    10,
    "oxx-":   -1,
    "-ooo-":   1e7,
    "-xxx-":  -1e6,
    "xooo-":   1e5,
    "oxxx-":  -1e4,
    "-oooo-":  1e11,
    "-xxxx-": -1e10,
    "xoooo-":  1e9,
    "oxxxx-": -1e8,
    "*ooooo*":  1e12,
    "*xxxxx*": -1e12,
}

def evaluate(state):
    featureCounter = extractFeatures(state)
    value = 0.0
    for feature in featureCounter.keys():
        value += 1.0 * featureCounter[feature] * weight[feature]
    return value

DEPTH = 6
GAMMA = 0.9
INFINITY = 1e14

class AlphaBetaAgent():
    def __init__(self, index=GameState.AI):
        self.index = index

    def value(self, state, depth, alpha, beta):
        who = state.next()
        depth += 1
        sign = 1.0 if who == state.first else -1.0
        if state.isLose(who):
            return -sign * INFINITY
        if state.isWin(who):
            return sign * INFINITY
        if depth == DEPTH:
            return evaluate(state)
        if who == state.first:
            return self.max_value(state, depth, alpha, beta)
        return self.min_value(state, depth, alpha, beta)

    def max_value(self, state, depth, alpha, beta):
        v = -INFINITY
        for action in self.suggestActions(state, depth):
            state.move(action)
            val = GAMMA * self.value(state, depth, alpha, beta)
            state.rewind()
            v = max(v, val)
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, state, depth, alpha, beta):
        v = INFINITY
        for action in self.suggestActions(state, depth):
            state.move(action)
            val = GAMMA * self.value(state, depth, alpha, beta)
            state.rewind()
            v = min(v, val)
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def suggestActions(self, state, depth):
        num = 15 - depth * 2
        legalActions = state.getLegalActions()
        if len(legalActions) < num:
            return legalActions
        reverse = state.next() == state.first
        def key(action):
            state.move(action)
            v = evaluate(state)
            state.rewind()
            return v
        return sorted(state.getLegalActions(), key=key, reverse=reverse)[:num]

    def getAction(self, state):
        v = -INFINITY if self.index == state.first else INFINITY
        ##### v Debug v #####
        # tracker = dict()
        ##### ^ Debug ^ #####
        suggestedActions = self.suggestActions(state, 0)
        best = choice(suggestedActions)        
        for action in suggestedActions:
            state.move(action)
            val = self.value(state, 0, -INFINITY, INFINITY)
            state.rewind()
            ##### v Debug v #####
            # tracker[action] = val
            ##### ^ Debug ^ #####
            if self.index == state.first and val > v:
                v = val
                best = action
            elif self.index != state.first and val < v:
                v = val
                best = action
        ##### v Debug v #####
        # print state
        # print "current value:", evaluate(state)
        # print "features:", extractFeatures(state)
        # print "optimal action:", best, "with value", v
        # print "other actions:", tracker
        ##### ^ Debug ^ #####
        return best

        