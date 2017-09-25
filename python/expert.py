""" parallel expert for dagger """

import sys
import numpy as np
import data_util as util
from state import State
from minimax import MinimaxAgent
from pathos.multiprocessing import ProcessPool

def demonstrate(games, parallel=True):
    def evaluate(game):
        s = State()
        a = MinimaxAgent(max_depth=6, max_width=6)
        ss = []
        pp = []
        for x, y in game:
            d = a.get_dist(s)
            if len(d) != 1 or (d[0][0] >= 0 and d[0][1] >= 0):
                ss.append(s.featurize())
                pp.append(util.dist_to_prob(d))
            s.move(x, y)
        sys.stdout.write("=")
        sys.stdout.flush()
        return (np.array(ss), np.array(pp))
    if parallel:
        pool = ProcessPool(nodes=7)
        results = list(pool.uimap(evaluate, games))
    else:
        results = list(map(evaluate, games))
    states = np.concatenate(list(map(lambda t: t[0], results)), axis=0)
    probs = np.concatenate(list(map(lambda t: t[1], results)), axis=0)
    return states, probs
