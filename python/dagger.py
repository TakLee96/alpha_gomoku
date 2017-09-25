""" DAgger training using minimax as expert """

from state import State
from agent import Agent
from resnet import export_meta
from minimax import MinimaxAgent
import sys
import numpy as np
import tensorflow as tf
import data_util as util

""" begin training """
with tf.Session() as sess:
    agent_mini = MinimaxAgent(max_depth=4, max_width=6)
    export_meta("dagger-model-1", "dagger")
    agent_conv = Agent(sess, "dagger-model-1", "dagger")

    for dagger_iter in range(500):
        states = []
        actions = []
        print("\nDagger Iter #%d" % dagger_iter)

        for game_iter in range(20):
            sys.stdout.write("=")
            sys.stdout.flush()
            s = State()
            while not s.end and len(s.history) < 30:
                dist = agent_mini.get_dist(s)
                if len(dist) != 1 or (dist[0][0] >= 0 and dist[0][1] >= 0):
                    states.append(s.featurize())
                    actions.append(util.dist_to_prob(dist))
                s.move(*agent_conv.get_action(s))
            s.save("dagger-games/%d-%d.pkl" % (dagger_iter, game_iter))
        print("\n  average game length: %f" % (len(states) / 20))

        X = np.array(states)
        Y = np.array(actions)
        loss1 = agent_conv.loss(X, Y)
        for _ in range(6):
            agent_conv.step(X, Y)
        loss2 = agent_conv.loss(X, Y)
        agent_conv.save(dagger_iter)
        print("  training completed and saved [loss %f -> %f]" % (loss1, loss2))
