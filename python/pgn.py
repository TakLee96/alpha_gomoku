from agent import Agent
from state import State
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("model1",  type=str)
parser.add_argument("chkpnt1", type=int)
parser.add_argument("model2",  type=str)
parser.add_argument("chkpnt2", type=int)
args = parser.parse_args()

graph1 = tf.Graph()
sess1  = tf.Session(graph=graph1)
with sess1.as_default():
    with graph1.as_default():
        agent1 = Agent(sess1, args.model1, chkpnt=args.chkpnt1)

graph2 = tf.Graph()
sess2  = tf.Session(graph=graph2)
with sess2.as_default():
    with graph2.as_default():
        agent2 = Agent(sess2, args.model2, chkpnt=args.chkpnt2)

for i in range(100):
    one_is_black = (i % 2) == 0
    if one_is_black:
        print("[White \"%s-%d\"]" % (args.model2, args.chkpnt2))
        print("[Black \"%s-%d\"]" % (args.model1, args.chkpnt1))
    else:
        print("[White \"%s-%d\"]" % (args.model1, args.chkpnt1))
        print("[Black \"%s-%d\"]" % (args.model2, args.chkpnt2))

    s = State()
    while not s.end and len(s.history) < 225:
        if one_is_black == (s.player > 0):
            with sess1.as_default():
                s.move(*agent1.get_safe_action(s))
        else:
            with sess2.as_default():
                s.move(*agent2.get_safe_action(s))
    if s.end:
        if s.player > 0:
            print("[Result \"0-1\"]")
        else:
            print("[Result \"1-0\"]")
    else:
        print("[Result \"1/2-1/2\"]")

    print()
    print("1.Nf3 Nf6")
    print()
