""" policy gradient """

from time import time
from agent import Agent
from state import State
import argparse
import numpy as np
import tensorflow as tf
import data_util as util

""" hyperparameters """
parser = argparse.ArgumentParser()
parser.add_argument("--iters", "-i", type=int, default=1000)
parser.add_argument("--samples", "-n", type=int, default=100)
parser.add_argument("--save", "-s", action="store_true")
args = parser.parse_args()

""" loading two models """
a_graph = tf.Graph()
b_graph = tf.Graph()
a_sess = tf.Session(graph=a_graph)
b_sess = tf.Session(graph=b_graph)
with a_sess.as_default():
    with a_graph.as_default():
        a_agent = Agent(a_sess, "supervised", "polygrad")
with b_sess.as_default():
    with b_graph.as_default():
        b_agent = Agent(b_sess, "supervised")

""" begin training """
for i in range(args.iters):
    X = []
    Y = []
    A = []
    num_win = 0
    t = time()
    for j in range(args.samples):
        s = State()
        a_is_black = (j % 2 == 0)
        while not s.end and len(s.history) < 225:
            if a_is_black == (s.player > 0):
                with a_sess.as_default():
                    with a_graph.as_default():
                        s.move(*a_agent.get_action(s))
            else:
                with b_sess.as_default():
                    with b_graph.as_default():
                        s.move(*b_agent.get_action(s))
        if args.save and j <= 1:
            s.save("games/%d-%d.pkl" % (i, int(a_is_black)))
        if s.end:
            score = float(s.player)
            if not a_is_black:
                score = -score
            num_win += (score > 0)
            moves = s.history
            s = State()
            for move in moves:
                if a_is_black == (s.player > 0):
                    X.append(s.featurize())
                    Y.append(util.one_hot(move))
                    A.append(score)
                s.move(*move)

    X = np.array(X)
    Y = np.array(Y)
    A = np.array(A)
    std = A.std()
    if std != 0:
        A = (A - A.mean()) / std

    with a_sess.as_default():
        with a_graph.as_default():
            loss1 = a_agent.pg_loss(X, Y, A)
            a_agent.pg_step(X, Y, A)
            loss2 = a_agent.pg_loss(X, Y, A)
            print("step %d loss %.04f -> %.04f" % (i, loss1, loss2))
            print("  polygrad wins %d/%d [%.04f sec]" % (num_win, args.samples, time() - t))
    if i % 10 == 0 and i > 0:
        with b_sess.as_default():
            b_graph = tf.Graph()
            with b_graph.as_default():
                b_agent = Agent(b_sess, "polygrad", random=True)
        print("! opponent switched to %s-%d" % (b_agent.model_name, b_agent.chkpnt))
    with a_sess.as_default():
        with a_graph.as_default():
            if i % 10 == 0:
                a_agent.save(i // 10)
                print("========== model saved at step %d ==========" % i)
