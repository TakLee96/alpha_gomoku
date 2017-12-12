import numpy as np
import pickle
import tensorflow as tf
from scipy.io import savemat
from state import State
from dual_agent import DualAgent

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()
ID = args.id

NUM_GAMES_GEN = 200
NUM_SIM = 100

X = []
V = []

with tf.Session() as sess:
    agent = DualAgent(sess, "treesup")
    for i in range(NUM_GAMES_GEN):
        print("sample %d" % i)
        s = State()
        while not s.end and len(s.history) < 225:
            s.move(*agent.get_safe_action(s))
        assert not s.violate, "we have a violated game"
        which = np.random.randint(len(s.history))
        h = s.history
        s = State()
        for k in range(which):
            s.move(*h[k])
        score = 0
        for j in range(NUM_SIM):
            t = s.copy()
            while not t.end and len(t.history) < 225:
                t.move(*agent.get_safe_action(t))
            assert not t.violate, "we have a violated game during simulation"
            score += t.player if t.end else 0
        score = 1.0 * score / NUM_SIM
        X.append(s.featurize())
        V.append(score)
        with open("value_treesup/%d-%d.pkl" % (ID, i), 'wb') as f:
            pickle.dump({"moves": s.history, "score": score, "violate": s.violate}, f)

X = np.array(X)
V = np.array(V)
savemat("value_treesup_%d" % ID, {"X": X, "V": V}, do_compression=True)
