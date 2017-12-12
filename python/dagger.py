""" DAgger training using minimax as expert """

from os import path
from time import time
from state import State
from agent import Agent
from policy_network import export_meta
from mcts_minimax_agent import MCTSMinimaxAgent
import sys
import numpy as np
import tensorflow as tf

""" hyperparameters """
DAGGER_ITERS = 500
NUM_GAMES = 2
NUM_SAMPLES = 100
SGD_STEPS = 50
NUM_TEST = 100

""" helper functions """
def one_hot(x, y):
    m = np.zeros(shape=226, dtype=np.float32)
    if x == -1 and y == -1:
        m[-1] = 1
    else:
        m[np.ravel_multi_index((x, y), dims=(15, 15))] = 1
    return m

""" begin training """
if not path.isdir("dagger"):
    export_meta("dagger")

init_iter = 0
current_graph = tf.Graph()
current_sess = tf.Session(graph=current_graph)
with current_sess.as_default():
    with current_graph.as_default():
        agent = MCTSMinimaxAgent(current_sess, "dagger")
        init_iter = agent.agent.chkpnt
        if init_iter == 0:
            agent.agent.save(0)

t_iter = time()
for dagger_iter in range(init_iter + 1, DAGGER_ITERS):
    with current_sess.as_default():
        print("=== Dagger Iter #%d ===" % dagger_iter)
        X = []
        Y = []

        for j in range(NUM_GAMES):
            s = State()
            while not s.end and len(s.history) < NUM_SAMPLES:
                s.move(*agent.agent.get_safe_action(s))
            s.save("games/%d.pkl" % dagger_iter)
            print("    sample game %d generated and saved, analyzing" % j)


            h = s.history
            s = State()
            agent.refresh()
            t = time()
            for x, y in h:
                X.append(s.featurize())
                Y.append(one_hot(*agent.get_action(s)))
                s.move(x, y)
                agent.update(s)
                sys.stdout.write("o")
                sys.stdout.flush()
            print()
            print("    sample game %d steps analyzed [%d sec]" % (len(h), time() - t))

        X = np.array(X)
        Y = np.array(Y)
        
        for i in range(SGD_STEPS+1):
            before = agent.agent.loss(X, Y)
            agent.agent.step(X, Y)
            after = agent.agent.loss(X, Y)
            if i % 10 == 0:
                print("    gradient descent: loss %.04f -> %.04f" % (before, after))
        print()

    prev_graph = tf.Graph()
    prev_sess = tf.Session(graph=prev_graph)
    with prev_sess.as_default():
        with prev_graph.as_default():
            prev_agent = Agent(prev_sess, "dagger")
    stat = np.zeros(shape=(2, 2), dtype=np.int)
    for i in range(NUM_TEST):
        s = State()
        prev_is_black = (i % 2 == 0)
        while not s.end and len(s.history) < 225:
            if prev_is_black == (s.player > 0):
                with prev_sess.as_default():
                    s.move(*prev_agent.get_safe_action(s))
            else:
                with current_sess.as_default():
                    s.move(*agent.agent.get_safe_action(s))
        sys.stdout.write("x")
        sys.stdout.flush()
        if s.end:
            stat[int(prev_is_black), int(s.player > 0)] += 1
    win_rate = (stat[0, 1] + stat[1, 0]) / stat.sum()
    print("    win_rate against old model is %.02f" % win_rate)
    if win_rate >= .55:
        agent.agent.save(dagger_iter)
        print("    new model %d saved" % dagger_iter)
    else:
        agent.agent.restore("dagger")
        print("    old model restored")
    prev_sess.close()
    del prev_sess, prev_graph
    print("    total time: %d sec\n" % (time() - t_iter))
    t_iter = time()
