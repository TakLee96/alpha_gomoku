import os
import pickle
import numpy as np
from state import State
from scipy.io import savemat

X = []
Y = []
V = []
stat = [0, 0, 0]
RESIGN = np.zeros(shape=226, dtype=np.float32)
RESIGN[-1] = 1

def one_hot(action):
    h = np.zeros(shape=226, dtype=np.float32)
    h[np.ravel_multi_index(action, dims=(15, 15))] = 1.0
    return h

class Children(dict):
    def __getitem__(self, key):
        if key not in self:
            self[key] = Node()
        return dict.__getitem__(self, key)

node_counter = 0
process_counter = 0

class Node:
    def __init__(self):
        global node_counter
        node_counter += 1
        self.children = Children()
        self.count = 0
        self.black_wins = 0
        self.best_move = None

    def dump(self, history, winner, index):
        if index >= len(history):
            return
        move = np.ravel_multi_index(history[index], dims=(15, 15))
        if index == len(history) - 1:
            self.best_move = move
        elif index == len(history) - 2:
            self.best_move = -1
        if winner == 1:
            self.black_wins += 1
        self.count += 1
        self.children[move].dump(history, winner, index+1)
    
    def compile(self, X, Y, V, s):
        global process_counter
        print("processing node: %d/%d" % (process_counter, node_counter))
        if self.count < 3:
            return  
        if self.best_move is None:
            best_moves = []
            best_rate = 0.0
            for move in self.children.keys():
                child = self.children[move]
                rate = 1.0 * child.black_wins / child.count
                rate = rate if s.player == 1 else 1.0 - rate
                if rate > best_rate:
                    best_rate = rate
                    best_moves = [move]
                elif rate == best_rate:
                    best_moves.append(move)
            if len(best_moves) == 0:
                best_moves = [-1]
            y = np.zeros(shape=226, dtype=np.float32)
            for move in best_moves:
                y[move] = 1
            y = y / y.sum()
            Y.append(y)
        else:
            y = np.zeros(shape=226, dtype=np.float32)
            y[self.best_move] = 1
            Y.append(y)
        X.append(s.featurize())
        v = 2 * (1.0 * self.black_wins / self.count - 0.5)
        V.append(v)
        process_counter += 1
        for move in self.children.keys():
            s.move(*np.unravel_index(move, dims=(15, 15)))
            self.children[move].compile(X, Y, V, s)
            s.rewind()

class Tree:
    def __init__(self):
        self.root = Node()

    def dump(self, history, winner):
        self.root.dump(history, winner, 0)

    def compile(self):
        X = []
        Y = []
        V = []
        s = State()
        self.root.compile(X, Y, V, s)
        return np.array(X), np.array(Y), np.array(V)


changes = [
    lambda t: (     t[0],      t[1]),    # original
    lambda t: (     t[1], 14 - t[0]),    # rotate 90
    lambda t: (14 - t[0], 14 - t[1]),    # rotate 180
    lambda t: (14 - t[1],      t[0]),    # rotate 270
    lambda t: (     t[0], 14 - t[1]),    # y-flip
    lambda t: (     t[1],      t[0]),    # rotate 90 + y-flip
    lambda t: (14 - t[0],      t[1]),    # rotate 180 + y-flip
    lambda t: (14 - t[1], 14 - t[0]),    # rotate 270 + y-flip
]


tree = Tree()
for name in os.listdir("minimax"):
    if os.path.isfile("minimax/" + name) and name.endswith(".pkl"):
        with open("minimax/" + name, "rb") as f:
            obj = pickle.load(f)
            moves = obj["history"]
            winner = obj["winner"]
            if winner == 1 or winner == -1:
                s = State()
                for i, move in enumerate(moves):
                    s.move(*move)
                assert s.end and not s.violate and s.player == winner
                for change in changes:
                    changed = list(map(change, moves))
                    tree.dump(changed, winner)
                stat[winner] += 1
                print("processed " + name)
            else:
                print("skipped " + name)

X, Y, V = tree.compile()
savemat("tree_minimax_mini", {"X": X, "Y": Y, "V": V}, do_compression=True)
print("black win %d white win %d even %d" % (stat[1], stat[-1], stat[0]))
