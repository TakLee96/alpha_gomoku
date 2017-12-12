""" supervised learning using minimax matches """

from state import State
from value_agent import ValueAgent
from scipy.io import loadmat
from value_network import export_meta
import numpy as np
import tensorflow as tf
import data_util as util

""" begin training """
with tf.Session() as sess:
    export_meta("valuenet")
    agent = ValueAgent(sess, "valuenet")
    matlab = loadmat("value_treesup")
    print("processing data")
    data = util.Data(matlab["X"], matlab["V"][0])
    print("processing complete")

    for i in range(1001):
        x_b, v_b = data.next_batch(400)
        lb = agent.l2_loss(x_b, v_b)
        agent.step(x_b, v_b)
        if i % 10 == 0:
            la = agent.l2_loss(x_b, v_b)
            x_b, v_b = data.test_batch(400)
            print("step %d train_l2_loss %.04f -> %.04f val_l2_loss %.04f reg_loss %.04f" % (i, lb, la, agent.l2_loss(x_b, v_b), agent.reg_loss(x_b, v_b)))
            # print("step %d train_l2_loss %.04f -> %.04f val_l2_loss %.04f" % (i, lb, la, agent.l2_loss(x_b, v_b)))
        if i % 50 == 0:
            agent.save(i)
            print("model saved at step %d" % i)
