""" policy gradient """

from agent import Agent
from state import State
import numpy as np
import tensorflow as tf
import data_util as util

""" hyperparameters """
PG_ITERS = 1000
NUM_TRAJECTORY = 50

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
for i in range(PG_ITERS):
    X = []
    Y = []
    A = []
    num_win = 0
    for j in range(NUM_TRAJECTORY):
        s = State()
        is_black = (j % 2 == 0)
        while not s.end and len(s.history) < 225:
            if is_black == (s.player > 0):
                with a_sess.as_default():
                    s.move(*a_agent.get_action(s))
            else:
                with b_sess.as_default():
                    s.move(*b_agent.get_action(s))
        if j <= 1:
            s.save("games/%d-%d.pkl" % (i, int(is_black)))
        if s.end:
            score = float(s.player)
            if not is_black:
                score = -score
            num_win += (score > 0)
            moves = s.history
            s = State()
            for move in moves:
                if is_black == (s.player > 0):
                    X.append(s.featurize())
                    Y.append(util.one_hot(move))
                    A.append(score)
                s.move(*move)

    X = np.array(X)
    Y = np.array(Y)
    A = np.array(A)
    #A = (A - A.mean()) / A.std()

    with a_sess.as_default():
        loss1 = a_agent.pg_loss(X, Y, A)
        a_agent.pg_step(X, Y, A)
        loss2 = a_agent.pg_loss(X, Y, A)
        print("step %d loss %.04f -> %.04f" % (i, loss1, loss2))
        print("  polygrad wins %d out of %d games" % (num_win, NUM_TRAJECTORY))

        if i % 10 == 0:
            a_agent.save(i)
            print("model saved at step %d" % i)
