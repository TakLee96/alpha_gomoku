""" battle between agents """
import sys
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from os import path
from state import State


changes = [
    lambda b: b,
    lambda b: b.T,
    lambda b: b[:, ::-1],
    lambda b: b[::-1, :],
    lambda b: b[::-1, ::-1],
    lambda b: b.T[:, ::-1],
    lambda b: b.T[::-1, :],
    lambda b: b.T[::-1, ::-1],
]
reverses = [
    lambda p: p,
    lambda p: p.T,
    lambda p: p[:, ::-1],
    lambda p: p[::-1, :],
    lambda p: p[::-1, ::-1],
    lambda p: p[:, ::-1].T,
    lambda p: p[::-1, :].T,
    lambda p: p[::-1, ::-1].T,
]


class Agent(mp.Process):
    def __init__(self, state_queue, action_queue, name, tensor_name):
        mp.Process.__init__(self)
        self.state_queue = state_queue
        self.action_queue = action_queue
        self.name = name
        self.tensor_name = tensor_name

    def run(self):
        with tf.Session() as session:
            saver = tf.train.import_meta_graph(path.join(path.dirname(__file__), "model", self.name, self.name + ".meta"), clear_devices=True)
            saver.restore(session, tf.train.latest_checkpoint(path.join(path.dirname(__file__), "model", self.name)))
            while True:
                state = self.state_queue.get()
                if state is None:
                    break
                else:
                    mean = np.zeros(shape=225, dtype=float)
                    for i in range(8):
                        prob = session.run(self.tensor_name, feed_dict={
                            "x:0": changes[i](state.player * state.board).reshape((1, 225)),
                            "y_:0": np.zeros(shape=(1, 225))}).reshape((15, 15))
                        prob = reverses[i](np.exp(prob - prob.max())) * (state.board == 0)
                        prob = prob / prob.sum()
                        mean += prob.reshape(225)
                    mean = mean / mean.sum()
                    action = np.unravel_index(np.random.choice(225, p=mean), (15, 15))
                    self.action_queue.put(action)


if __name__ == "__main__":
    states = [None, mp.Queue(), mp.Queue()]
    names = [None, "deepsdknet", "godsdknet"]
    print("black: " + names[1])
    print("white: " + names[-1])
    actions = mp.Queue()
    players = [Agent(states[i], actions, names[i], "y:0") for i in [1, -1]]
    for player in players:
        player.start()
    win_count = [0, 0, 0]
    end_symbl = ["v", "o", "x"]
    for _ in range(100):
        state = State()
        while not state.end and len(state.history) != 225:
            states[state.player].put(state)
            x, y = actions.get()
            state.move(x, y)
            sys.stdout.write(".")
        if state.end:
            win_count[state.player] += 1
            print(end_symbl[state.player])
        else:
            win_count[0] += 1
            print(end_symbl[0])
    states[1].put(None)
    states[-1].put(None)
    print("black wins %d games" % win_count[1])
    print("white wins %d games" % win_count[-1])
    print("there are %d draws" % win_count[0])
