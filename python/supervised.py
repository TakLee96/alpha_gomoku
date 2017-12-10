""" supervised learning using minimax matches """

from state import State
from agent import Agent
from scipy.io import loadmat
from policy_network import export_meta
import numpy as np
import tensorflow as tf
import data_util as util

""" begin training """
with tf.Session() as sess:
    export_meta("supervised")
    agent = Agent(sess, "supervised")
    matlab = loadmat("minimax")
    print("processing data")
    data = util.Data(matlab["X"], matlab["Y"])
    print("processing complete")

    for i in range(4001):
        x_b, y_b = data.next_batch(256)
        agent.step(x_b, y_b)
        if i % 10 == 0:
            x_b, y_b = data.test_batch(1024)
            print("step %d loss %.04f accuracy %.04f" % (i, agent.loss(x_b, y_b), agent.accuracy(x_b, y_b)))
        if i % 100 == 0:
            agent.save(i)
            print("model saved at step %d" % i)
