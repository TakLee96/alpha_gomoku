""" battle arena between agents """

import argparse
import numpy as np
import tensorflow as tf
from time import time
from agent import Agent
from state import State

parser = argparse.ArgumentParser()
parser.add_argument("model_name_1", type=str)
parser.add_argument("model_name_2", type=str)
parser.add_argument("--chkpnt1", "-c1", type=int)
parser.add_argument("--chkpnt2", "-c2", type=int)
parser.add_argument("--num_games", "-n", default=100, type=int)
parser.add_argument('--deterministic', '-d', action='store_true')
parser.add_argument('--save', '-s', action='store_true')
args = parser.parse_args()

a_graph = tf.Graph()
b_graph = tf.Graph()
a_sess = tf.Session(graph=a_graph)
b_sess = tf.Session(graph=b_graph)
with a_sess.as_default():
    with a_graph.as_default():
        a_agent = Agent(a_sess, args.model_name_1, chkpnt=args.chkpnt1)
with b_sess.as_default():
    with b_graph.as_default():
        b_agent = Agent(b_sess, args.model_name_2, chkpnt=args.chkpnt2)
print("ARENA: %s-%d VERSES %s-%d" % (a_agent.model_name, a_agent.chkpnt, b_agent.model_name, b_agent.chkpnt))

stat = np.zeros(shape=(2, 2), dtype=np.int)
for i in range(args.num_games):
    t = time()
    s = State()
    a_is_black = (i % 2 == 0)
    while not s.end and len(s.history) < 225:
        if a_is_black == (s.player > 0):
            with a_sess.as_default():
                s.move(*a_agent.get_action(s, deterministic=args.deterministic))
        else:
            with b_sess.as_default():
                s.move(*b_agent.get_action(s, deterministic=args.deterministic))
    if len(s.history) == 225:
        print("UNBELIEVABLE EVEN!")
    stat[int(a_is_black), int(s.player > 0)] += 1
    print("match %d winner %s [%.04f sec]" %
        (i, (a_agent.model_name if a_is_black == (s.player > 0) else b_agent.model_name), time() - t))
    if args.save:
        s.save("arena/%d.pkl" % i)

print("Of the %d games between them" % args.num_games)
print("  %s as black wins %d" % (a_agent.model_name, stat[1, 1]))
print("  %s as white wins %d" % (a_agent.model_name, stat[0, 0]))
print("  %s overall wins %d" % (a_agent.model_name, stat[0, 0] + stat[1, 1]))
print("  %s against %s overall win-rate %f" %
    (a_agent.model_name, b_agent.model_name, (stat[1, 1] + stat[0, 0]) / stat.sum()))
