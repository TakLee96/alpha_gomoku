import os
import pickle
import numpy as np
import data_util as util
from state import State
from scipy.io import savemat

X = []
Y = []
stat = [0, 0, 0]

for name in os.listdir("minimax"):
    if os.path.isfile("minimax/" + name) and name.endswith(".pkl"):
        with open("minimax/" + name, "rb") as f:
            obj = pickle.load(f)
            moves = obj["history"]
            winner = obj["winner"]
            s = State()
            for move in moves:
                if s.player == winner:
                    X.append(s.featurize())
                    Y.append(util.one_hot(move))
                s.move(*move)
            stat[winner] += 1
            print("processed " + name)

savemat("minimax", {"X": X, "Y": Y}, do_compression=True)
print("black win %s white win %s" % (stat[1], stat[-1]))
