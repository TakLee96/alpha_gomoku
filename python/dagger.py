""" DAgger training using minimax as expert """

from time import time
from state import State
from agent import Agent
from expert import demonstrate
from minimax import MinimaxAgent
from policy_network import export_meta
import numpy as np
import tensorflow as tf

""" hyperparameters """
DAGGER_ITERS = 500
GAME_ITERS = 21
MAX_GAME_LENGTH = 30
SGD_STEPS = 6

""" begin training """
with tf.Session() as sess:
    export_meta("dagger", "dagger")
    agent = Agent(sess, "dagger", "dagger")
    initials = [(4, 4), (4, 7), (4, 11), (7, 4), (7, 7), (7, 11), (11, 4), (11, 7), (11, 11)]

    for dagger_iter in range(DAGGER_ITERS):
        print("Dagger Iter #%d" % dagger_iter)
        games = []
        for game_iter in range(GAME_ITERS):
            s = State()
            s.move(*initials[np.random.randint(len(initials))])
            while not s.end and len(s.history) < MAX_GAME_LENGTH:
                s.move(*agent.get_action(s))
            games.append(s.history)
            s.save("games/%d-%d.pkl" % (dagger_iter, game_iter))
        
        t = time()
        X, Y = demonstrate(games)
        print("\n  %d games analyzed [%.02f sec]" % (GAME_ITERS, time() - t))
        print("  average game length: %f" % (len(X) / GAME_ITERS))

        t = time()
        loss1 = agent.loss(X, Y)
        for _ in range(SGD_STEPS):
            agent.step(X, Y)
        loss2 = agent.loss(X, Y)
        agent.save(dagger_iter)
        print("  training completed and saved [loss %f -> %f]\n" % (loss1, loss2))
