from agent import Agent
from scipy.io import loadmat
from dual_agent import DualAgent
from dual_network import export_meta

import numpy as np
import tensorflow as tf
import data_util as util


""" begin training """
with tf.Session() as sess:
    export_meta("treesup")
    agent = DualAgent(sess, "treesup")
    matlab = loadmat("tree_minimax")
    print("processing data")
    data = util.TenaryData(matlab["X"], matlab["Y"], matlab["V"][0])
    print("processing complete")

    for i in range(8001):
        x_b, y_b, v_b = data.next_batch(1024)
        if i % 10 == 0:
            x_b, y_b, v_b = data.test_batch(1024)
            pl, vl, rl, l = agent.loss(x_b, y_b, v_b)
            print("\nstep %d analysis" % i)
            print(">>> policy loss: %f" % pl)
            print(">>> value loss: %f" % vl)
            print(">>> regularization loss: %f" % rl)
            print(">>> loss: %f" % l)
            print(">>> accuracy: %f" % agent.accuracy(x_b, y_b))

        agent.step(x_b, y_b, v_b)
        if i % 100 == 0:
            agent.save(i)
            print("model saved at step %d" % i)
