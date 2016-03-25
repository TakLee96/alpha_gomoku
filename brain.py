from random import choice
from time import time

from game import GameState
from evaluate import extractFeatures, evaluate, characterize, INFINITY

current_time_millis = lambda: int(round(time() * 1000))

def randomAction(state):
    return choice(state.getLegalActions())

def counterDeadFour(state):
    who = state.next()
    moves = list()
    for move in state.getLegalActions():
        state.move(move)
        features = extractFeatures(state, who)
        state.rewind()
        if '*ooooo*' in features:
            return [move]
        if not dead_four(features):
            moves.append(move)
    return moves

def counterLiveThree(state):
    who = state.next()
    moves = list()
    for move in state.getLegalActions():
        state.move(move)
        features = extractFeatures(state, who)
        state.rewind()
        if gonna_win(features): return [move]
        if not live_three(features) or my_four(features):
            moves.append(move)
    return moves

def top_moves(tuples, num):
    if len(tuples) > num:
        tuples = sorted(tuples, key=lambda t: t[1], reverse=True)[:num]
    return map(lambda t: t[0], tuples)

def bestGrowthMoves(state, curr_features, num=5):
    who = state.next()
    styles = {'atk': [], 'def': [], 'ntr': []}
    for move in state.getLegalActions():
        state.move(move)
        features = extractFeatures(state, who)
        state.rewind()
        if gonna_win(features): return [move]
        char = characterize(curr_features, features)
        styles[char].append( (move, evaluate(features)) )
    for style in styles.keys():
        styles[style] = top_moves(styles[style], num)
    return reduce(lambda a, b: a + b, styles.values())

def winTheGame(state):
    who = state.next()
    for move in state.getLegalActions():
        state.move(move)
        if state.isWin(who):
            state.rewind()
            return [move]
        state.rewind()
    assert False

def my_four(features):
    return ('xoooo-'  in features or
            '*o-ooo*' in features or
            '*ooo-o*' in features or
            '*oo-oo*' in features or
            '-oooo-'  in features)

def dead_four(features):
    return ('oxxxx-'  in features or
            '*x-xxx*' in features or
            '*xxx-x*' in features or
            '*xx-xx*' in features or
            'long-death' in features)

def live_three(features):
    return ('-xxx-'  in features or
            '-x-xx-' in features or
            '-xx-x-' in features)

def gonna_win(features):
    return ('*ooooo*' in features or
            '-oooo-'  in features)

DETER_DEPTH = 3
DEPTH = 8

class UNTSAgent():
    """ Unbalanced Non-zero-sum Tree Search Agent """
    def __init__(self, index=GameState.AI):
        self.index = index
    
    def value(self, state, depth, deter_depth):
        who = state.next()
        if state.isLose(who):
            return (-INFINITY, INFINITY)
        if state.isWin(who):
            return (INFINITY, -INFINITY)
        if depth == DEPTH or deter_depth == DETER_DEPTH:
            return (None, None)

        moves, deter = self.getActions(state)
        depth += 1
        deter_depth += deter
        if len(moves) == 0:
            moves = [randomAction(state)]
        def val(move):
            who = state.next()
            state.move(move)
            oppval, myval = self.value(state, depth, deter_depth)
            if myval is None:
                myval = evaluate(extractFeatures(state, who))
            state.rewind()
            return (myval, oppval)
        return max(map(val, moves), key=lambda t: t[0])

    def getActions(self, state, features=None):
        features = features or extractFeatures(state, state.next())
        moves_to_status = dict()
        if my_four(features):
            return winTheGame(state), False
        if '-xxxx-' in features:
            return list(), False
        if dead_four(features):
            return counterDeadFour(state), False
        if live_three(features):
            return counterLiveThree(state), False
        return bestGrowthMoves(state, features), True

    def getAction(self, state):
        if len(state.hist) == 0:
            return (GameState.GRID_SIZE/2, GameState.GRID_SIZE/2)
        features = extractFeatures(state, state.next())
        print state
        print features
        moves, deter = self.getActions(state, features)
        if len(moves) == 0:
            print "[RESULT] AI thinks he is gonna lose"
            return randomAction(state)
        elif len(moves) == 1:
            print "[RESULT] AI thinks it's obviously here:", moves[0]
            return moves[0]
        def key(move):
            state.move(move)
            oppval, myval = self.value(state, 1, deter)
            state.rewind()
            print move, myval, "|",
            return myval
        print "[RESULT] AI is thinking"
        start = current_time_millis()
        best = max(moves, key=key)
        end = current_time_millis()
        print "\nAI thinks it should be here:", best, "[" + str(end-start) + "ms]"
        return best
        