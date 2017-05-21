import sys
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from os import path
from gomoku import State


class Agent(mp.Process):
    def __init__(self, state_queue, action_queue, name, tensor_name, player):
        mp.Process.__init__(self)
        self.state_queue = state_queue
        self.action_queue = action_queue
        self.name = name
        self.tensor_name = tensor_name
        self.player = player

    def run(self):
        with tf.Session() as session:
            saver = tf.train.import_meta_graph(path.join(path.dirname(__file__), "model", self.name, self.name + ".meta"), clear_devices=True)
            saver.restore(session, tf.train.latest_checkpoint(path.join(path.dirname(__file__), "model", self.name)))
            while True:
                state = self.state_queue.get()
                if state is None:
                    break
                elif state.player == self.player:
                    prob = session.run(self.tensor_name, feed_dict={
                        "x:0": state.board.reshape((1, 225)),
                        "y_:0": np.zeros(shape=(1, 225))}).reshape((15, 15))
                    prob = np.exp(prob - prob.max()) * (state.board == 0)
                    prob = (prob / prob.sum()).reshape(225)
                    action = np.unravel_index(np.random.choice(225, p=prob), (15, 15))
                    self.action_queue.put(action)
                else:
                    self.state_queue.put(state)


if __name__ == "__main__":
    states = mp.Queue()
    actions = mp.Queue()
    players = [
        Agent(states, actions, "sdknet", "Reshape_1:0", 1),
        Agent(states, actions, "deepsdknet", "y:0", -1),
    ]
    for player in players:
        player.start()
    win_count = [0, 0, 0]
    end_symbl = ["v", "o", "x"]
    for _ in range(100):
        state = State()
        while not state.end and len(state.history) != 225:
            states.put(state)
            x, y = actions.get()
            state.move(x, y)
            sys.stdout.write(".")
        if state.end:
            win_count[state.player] += 1
            print(end_symbl[state.player])
        else:
            win_count[0] += 1
            print(end_symbl[0])
    for _ in range(2):
        states.put(None)
    print("black wins %d games" % win_count[1])
    print("white wins %d games" % win_count[-1])
    print("there are %d draws" % win_count[0])
