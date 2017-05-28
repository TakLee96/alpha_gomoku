""" battle between agents """
import sys
import pickle
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from os import path
from state import State
from feature import diff


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
            root = path.join(path.dirname(__file__), "model", "policy" self.name)
            saver = tf.train.import_meta_graph(path.join(root, self.name + ".meta"), clear_devices=True)
            saver.restore(session, tf.train.latest_checkpoint(root))
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
                    if state.player == 1:
                        for i in range(15):
                            for j in range(15):
                                new = diff(state, i, j)
                                if new["-o-oo-"] + new["-ooo-"] >= 2 or \
                                    new["four-o"] + new["-oooo-"] >= 2 or state._long(i, j):
                                    mean[i*15+j] = 0
                    mean = mean / mean.sum()
                    action = np.unravel_index(np.random.choice(225, p=mean), (15, 15))
                    self.action_queue.put(action)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] not in ("fight", "save"):
        print("Usage: python battle.py [fight/save] [iterations] [black] [white]")
    else:
        states = [None, mp.Queue(), mp.Queue()]
        names = [None, sys.argv[3], sys.argv[4]]
        print("black: " + names[1])
        print("white: " + names[-1])
        actions = mp.Queue()
        players = [Agent(states[i], actions, names[i], "y:0") for i in [1, -1]]
        for player in players:
            player.start()
        win_count = [0, 0, 0]
        end_symbl = ["v", "o", "x"]
        for j in range(int(sys.argv[2])):
            state = State()
            sys.stdout.write("%.4d " % j)
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
            if sys.argv[1] == "save":
                with open(path.join(path.dirname(__file__), "data", names[1], "%.4d.pkl" % j), "wb") as out:
                    pickle.dump({
                        "history": state.history,
                        "winner": state.player if state.end else 0 }, out)

        states[1].put(None)
        states[-1].put(None)
        print("black: " + names[1])
        print("white: " + names[-1])
        print("black wins %d games" % win_count[1])
        print("white wins %d games" % win_count[-1])
        print("there are %d draws" % win_count[0])
