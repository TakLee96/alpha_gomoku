import pickle
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from time import time
from state import State
from sys import argv, exit
from itertools import count
from os import path, listdir
from minimax import MinimaxAgent


available = ("minimax", "monet")
if len(argv) != 3 or argv[1] not in available or argv[2] not in available:
    print("Usage: python3 battle_new.py [agent] [agent]")
    exit()


class NetAgent(mp.Process):
    def __init__(self, which):
        mp.Process.__init__(self)
        self.state_queue = mp.Queue()
        self.dist_queue = mp.Queue()
        self.name = "monet-" + which
        self.start()

    def run(self):
        with tf.Session() as session:
            root = path.join(path.dirname(__file__), "model", "policy", self.name)
            saver = tf.train.import_meta_graph(path.join(root, self.name + ".meta"), clear_devices=True)
            saver.restore(session, tf.train.latest_checkpoint(root))
            while True:
                state = self.state_queue.get()
                if state is None:
                    return
                board = np.ndarray(shape=(1, 15, 15, 2), dtype=np.float32)
                board[:, :, :, 0] = (state.board > 0)
                board[:, :, :, 1] = (state.board < 0) 
                y = session.run("y:0", feed_dict={
                    "y_:0": np.zeros(shape=(1, 225)),
                    "x:0": board }).reshape(225)
                y = np.exp(y)
                y = y / y.sum()
                self.dist_queue.put(y)

    def get_action(self, state):
        if len(state.history) == 0:
            return (7, 7)
        self.state_queue.put(state)
        prob = self.dist_queue.get()
        return np.unravel_index(np.random.choice(225, p=prob), dims=(15, 15))


agents = {
    "minimax": lambda which: MinimaxAgent(max_depth=6, max_width=8),
    "monet": lambda which: NetAgent(which),
}
players = [0, agents[argv[1]]("black"), agents[argv[2]]("white")]


names = ["", "black", "white"]
state = State()
while len(state.history) != 225 and not state.end:
    t = time()
    x, y = players[state.player].get_action(state)
    print("%s [%g seconds]" % (names[state.player], time() - t))
    state.move(x, y)
winner = state.player if state.end else 0
with open(path.join(path.dirname(__file__), "data", "battle-%s-%s.pkl" % (argv[1], argv[2])), "wb") as out:
    pickle.dump({"history": state.history, "winner": winner }, out)
print("winner: %d" % winner)
exit()
