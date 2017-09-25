""" generate data for value network """

from agent import Agent

with tf.Session() as sess:
    agent = Agent(sess)
    for i in range(100):
        # TODO