""" DAgger training using MCTS as expert """

import sys
import numpy as np
import tensorflow as tf
from time import time
from state import State
from dual_agent import DualAgent
from minimax_network import MinimaxNetworkAgent
from dual_network import export_meta
from data_util import TenaryOnlineData, one_hot


NUM_DAGGER_ITER = 1024
NUM_GAMES_PER_DAGGER = 5
NUM_LABEL_MOVE_PER_GAME = 128
NUM_SGD_ITER = 1
SIZE_BATCH = 640
NUM_TEST_GAMES = 11


def get_random_action(state):
    y_p = DualAgent.get_dist(agent, state)
    c = np.random.choice(225, p=y_p)
    x, y = np.unravel_index(c, dims=(15, 15))
    assert state.board[x, y] == 0, "total prob %f that prob %f" % (y_p.sum(), y_p[c])
    return x, y

def get_mcts_action_and_value(state):
    dst = agent.get_dist(state)
    val = agent.get_value(state)
    count = np.zeros(shape=226, dtype=np.float32)
    if state.player * val < -0.9:
        count[-1] = 1
    else:
        count[:-1] = dst * dst
    return count / count.sum(), val

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
    agent.refresh()
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
    return X, Y, V


def arena():
    global better
    prev_graph = tf.Graph()
    prev_sess = tf.Session(graph=prev_graph)
    if not better:
        prev = "treesup"
    else:
        prev = "dualdagger"
    with prev_sess.as_default():
        with prev_graph.as_default():
            prev_agent = MCTSAgent(prev_sess, prev)

    stat = np.zeros(shape=(2, 2), dtype=np.int)
    for i in range(NUM_TEST_GAMES):
        agent.refresh()
        prev_agent.refresh()
        s = State()
        prev_is_black = (i % 2 == 0)
        while not s.end and len(s.history) < 225:
            if prev_is_black == (s.player > 0):
                with prev_sess.as_default():
                    s.move(*get_random_mcts_action(prev_agent, s))
            else:
                with current_sess.as_default():
                    s.move(*get_random_mcts_action(agent, s))
            agent.update(s)
            prev_agent.update(s)
        if len(s.history) == 225:
            print("UNBELIEVABLE EVEN!")
        sys.stdout.write("x")
        sys.stdout.flush()
        stat[int(prev_is_black), int(s.player > 0)] += 1
    win_rate = (stat[0, 1] + stat[1, 0]) / stat.sum()
    print("\nwin_rate is %.02f" % win_rate)
    if win_rate > .5:
        better = True
        agent.save(iteration)
        print("new model %d saved" % iteration)
    else:
        agent.restore(prev)
        print("old model restored")
    prev_sess.close()


current_graph = tf.Graph()
current_sess = tf.Session(graph=current_graph)
with current_sess.as_default():
    with current_graph.as_default():
        agent = MinimaxNetworkAgent(current_sess, "treesup", "dualdagger")
        print("Initialization complete")


better = False
for iteration in range(NUM_DAGGER_ITER):
    with current_sess.as_default():
        t = time()
        print("\nDAgger iteration %d" % iteration)
        data = TenaryOnlineData()
        for j in range(NUM_GAMES_PER_DAGGER):
            game = generate_game(save=(j==0), iter=iteration)
            sys.stdout.write("+")
            sys.stdout.flush()
            data.store(*analyze_game(game))
            print()
        
        for j in range(NUM_SGD_ITER):
            agent.step(*data.next_batch(SIZE_BATCH))
        print("training complete")

    arena()
    print("== time: %.02f seconds" % (time() - t))

