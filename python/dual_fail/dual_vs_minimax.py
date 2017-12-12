""" battle arena between agents """

import argparse
import numpy as np
import tensorflow as tf
from time import time
from state import State
from minimax import MinimaxAgent
from mcts_agent import MCTSAgent

NUM_GAMES = 2

with tf.Session() as sess:
    mcts = MCTSAgent(sess, "dualsup", chkpnt=3000)
    agent = MinimaxAgent()
    print("ARENA: %s-%d VERSES %s-%d" % (mcts.model_name, mcts.chkpnt, "minimax", 0))

    stat = np.zeros(shape=(2, 2), dtype=np.int)
    for i in range(NUM_GAMES):
        t = time()
        s = State()
        a_is_black = (i % 2 == 0)
        while not s.end and len(s.history) < 225:
            if a_is_black == (s.player > 0):
                s.move(*mcts.get_action(s, deterministic=True))
                mcts.update(s)
            else:
                s.move(*agent.get_action(s))
                mcts.update(s)
        mcts.refresh()
        if len(s.history) == 225:
            print("UNBELIEVABLE EVEN!")
        stat[int(a_is_black), int(s.player > 0)] += 1
        print("match %d winner %s [%.04f sec]" %
            (i, ("a-dualsup" if a_is_black == (s.player > 0) else "b-minimax"), time() - t))
        s.save("arena/%d.pkl" % i)

    print("Of the %d games between them" % NUM_GAMES)
    print("  %s as black wins %d" % (mcts.model_name, stat[1, 1]))
    print("  %s as white wins %d" % (mcts.model_name, stat[0, 0]))
    print("  %s overall wins %d" % (mcts.model_name, stat[0, 0] + stat[1, 1]))
    print("  %s against %s overall win-rate %f" %
        (mcts.model_name, "minimax", (stat[1, 1] + stat[0, 0]) / stat.sum()))
