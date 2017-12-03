""" DAgger training using MCTS as expert """

import sys
import numpy as np
import tensorflow as tf
from time import time
from state import State
from dual_agent import DualAgent
from mcts_agent import MCTSAgent
from dual_network import export_meta
from data_util import TenaryOnlineData, one_hot


NUM_DAGGER_ITER = 64
NUM_GAMES_PER_DAGGER = 64
NUM_LABEL_MOVE_PER_GAME = 32
NUM_SGD_ITER = 64
SIZE_BATCH = 128


def get_random_action(state):
    y_p = DualAgent.get_dist(agent, state)
    c = np.random.choice(225, p=y_p)
    x, y = np.unravel_index(c, dims=(15, 15))
    assert state.board[x, y] == 0, "total prob %f that prob %f" % (y_p.sum(), y_p[c])
    return x, y

def get_mcts_action_and_value(state):
    v = agent.get_value(state)
    count = np.zeros(shape=226, dtype=np.float32)
    if state.player * v < -0.9:
        count[-1] = 1
    else:
        count[:-1] = agent.get_dist(state)
        count[count != count.max()] = 0
    return count / count.sum(), v

def generate_game(save=False, iter=0):
    """ generate a game played by the neural network  """
    s = State()
    while not s.end and len(s.history) < 225:
        x, y = get_random_action(s)
        s.move(x, y)
    if save:
        s.save("dualdagger/game-%d.pkl" % iter)
    return s.history

def analyze_game(game):
    """ generate expert predictions on moves and values """
    s = State()
    X = []
    Y = []
    V = []
    if len(game) > NUM_LABEL_MOVE_PER_GAME:
        which = set(np.random.choice(len(game), NUM_LABEL_MOVE_PER_GAME, replace=False))
        for i, (x, y) in enumerate(game):
            if i in which:
                X.append(s.featurize())
                yp, vp = get_mcts_action_and_value(s)
                Y.append(yp)
                V.append(vp)
                sys.stdout.write("o")
                sys.stdout.flush()
            s.move(x, y)
            agent.refresh()
    else:
        for x, y in game:
            X.append(s.featurize())
            yp, vp = get_mcts_action_and_value(s)
            Y.append(yp)
            V.append(vp)
            s.move(x, y)
            agent.update(s)
            sys.stdout.write("o")
            sys.stdout.flush()
    agent.refresh()
    return X, Y, V

# export_meta("dualdagger")

with tf.Session() as sess:
    agent = MCTSAgent(sess, "treesup", "dualdagger", 5000)
    print("Initialization complete")

    for i in range(NUM_DAGGER_ITER):
        t = time()
        print("\nDAgger iteration %d" % i)
        data = TenaryOnlineData()
        for j in range(NUM_GAMES_PER_DAGGER):
            game = generate_game(save=(j==0), iter=i)
            sys.stdout.write("+")
            sys.stdout.flush()
            data.store(*analyze_game(game))
            print()
        
        for j in range(NUM_SGD_ITER):
            agent.step(*data.next_batch(SIZE_BATCH))
        print("training complete")
        agent.save(i)
        print("saving %d complete" % i)
        print("== time: %.02f seconds" % (time() - t))
