import os
import pickle
import numpy as np
import data_util as util
from state import State
from scipy.io import savemat

X = []
Y = []
stat = [0, 0, 0]
RESIGN = np.zeros(shape=226, dtype=np.float32)
RESIGN[-1] = 1

def one_hot(action):
    h = np.zeros(shape=226, dtype=np.float32)
    h[np.ravel_multi_index(action, dims=(15, 15))] = 1.0
    return h

for name in os.listdir("minimax"):
    if os.path.isfile("minimax/" + name) and name.endswith(".pkl"):
        with open("minimax/" + name, "rb") as f:
            obj = pickle.load(f)
            moves = obj["history"]
            winner = obj["winner"]
            s = State()
            for i, move in enumerate(moves):
                X.append(s.featurize())
                if i == len(moves) - 2 and s.player == -winner:
                    Y.append(RESIGN)
                else:
                    Y.append(one_hot(move))
                s.move(*move)
            stat[winner] += 1
            print("processed " + name)

savemat("minimax", {"X": X, "Y": Y}, do_compression=True)
print("black win %s white win %s" % (stat[1], stat[-1]))
