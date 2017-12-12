""" battle arena between agents """

import argparse
import numpy as np
import tensorflow as tf
from sys import stdout
from time import time
from state import State
from mcts_agent import MCTSAgent

parser = argparse.ArgumentParser()
parser.add_argument("model_name_1", type=str)
parser.add_argument("model_name_2", type=str)
parser.add_argument("--epsilon1", "-e1", type=float)
parser.add_argument("--epsilon2", "-e2", type=float)
parser.add_argument("--multiplier1", "-m1", type=float)
parser.add_argument("--multiplier2", "-m2", type=float)
parser.add_argument("--chkpnt1", "-c1", type=int)
parser.add_argument("--chkpnt2", "-c2", type=int)
parser.add_argument("--num_games", "-n", default=100, type=int)
parser.add_argument('--save', '-s', action='store_true')
args = parser.parse_args()

a_graph = tf.Graph()
b_graph = tf.Graph()
a_sess = tf.Session(graph=a_graph)
b_sess = tf.Session(graph=b_graph)
with a_sess.as_default():
    with a_graph.as_default():
        a_agent = MCTSAgent(a_sess, args.model_name_1, chkpnt=args.chkpnt1, epsilon=args.epsilon1, multiplier=args.multiplier1)
with b_sess.as_default():
    with b_graph.as_default():
        b_agent = MCTSAgent(b_sess, args.model_name_2, chkpnt=args.chkpnt2, epsilon=args.epsilon2, multiplier=args.multiplier2)
stdout.write("A(e=%.01f, m=%.01f) vs B(e=%.01f, m=%.01f) = " % (args.epsilon1, args.multiplier1, args.epsilon2, args.multiplier2))
stdout.flush()

stat = np.zeros(shape=(2, 2), dtype=np.int)
for i in range(args.num_games):
    t = time()
    s = State()
    a_is_black = (i % 2 == 0)
    while not s.end and len(s.history) < 225:
        if a_is_black == (s.player > 0):
            with a_sess.as_default():
                s.move(*a_agent.get_action(s, deterministic=True))
                a_agent.update(s)
                b_agent.update(s)
        else:
            with b_sess.as_default():
                s.move(*b_agent.get_action(s, deterministic=True))
                a_agent.update(s)
                b_agent.update(s)
    a_agent.refresh()
    b_agent.refresh()
    if len(s.history) == 225:
        print("UNBELIEVABLE EVEN!")
    stat[int(a_is_black), int(s.player > 0)] += 1
    if args.save:
        s.save("arena/%d.pkl" % i)

if stat[1, 1] + stat[0, 0] == 2:
    stdout.write("A\n")
elif stat[1, 1] + stat[0, 0] == 0:
    stdout.write("B\n")
else:
    stdout.write("?\n")
stdout.flush()
