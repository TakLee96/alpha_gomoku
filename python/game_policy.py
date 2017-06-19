""" battle between agent and human """
import numpy as np
import tkinter as tk
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


class Application(tk.Frame):
    def __init__(self, state_queue, dist_queue, master):
        tk.Frame.__init__(self, master)
        self.button = list()
        self.frames = list()
        self.state = State()
        root = path.join(path.dirname(__file__), "img")
        self.image = [
          tk.PhotoImage(file=path.join(root, "empty.gif")),
          tk.PhotoImage(file=path.join(root, "naught.gif")),
          tk.PhotoImage(file=path.join(root, "cross.gif")),
        ]
        self.state_queue = state_queue
        self.dist_queue = dist_queue
        self.pack()
        self.create_widgets()
        self.draw_probability()

    def draw_probability(self):
        if self.state.end:
            return
        self.state_queue[self.state.player].put(self.state)
        dist = self.dist_queue.get()
        maxp = dist.max()
        for i in range(15):
            for j in range(15):
                button = self.button[np.ravel_multi_index((i, j), dims=(15, 15))]
                if dist[i, j] == maxp:
                    button.config(bg="red", image="", text="%.2f" % dist[i, j])
                elif dist[i, j] > 0.01:
                    button.config(bg="yellow", image="", text="%.2f" % dist[i, j])
                else:
                    button.config(bg=None, image=self.image[self.state.board[i, j]])

    def highlight(self, x, y):
        for i, j in self.state.highlight(x, y):
            self.frames[np.ravel_multi_index((i, j), dims=(15, 15))].config(padx=1, pady=1, bg="blue")

    def click(self, i, j):
        def respond(e):
            if not self.state.end and self.state.board[i, j] == 0:
                self.button[np.ravel_multi_index((i, j), dims=(15, 15))].config(image=self.image[self.state.player])
                self.state.move(i, j)
                if self.state.end:
                    if self.state.features["win-o"] + self.state.features["win-x"] > 0:
                        self.highlight(i, j)
                    else:
                        self.frames[np.ravel_multi_index((i, j), dims=(15, 15))].config(padx=1, pady=1, bg="red")
                else:
                    self.draw_probability()
        return respond

    def create_widgets(self):
        for i in range(15):
            for j in range(15):
                f = tk.Frame(self, height=50, width=50)
                f.pack_propagate(0)
                f.grid(row=i, column=j, padx=0, pady=0)
                self.frames.append(f)
                b = tk.Label(f, image=self.image[0])
                b.pack(fill=tk.BOTH, expand=1)
                b.bind("<Button-1>", self.click(i, j))
                self.button.append(b)


class Agent(mp.Process):
    def __init__(self, state_queue, dist_queue, name):
        mp.Process.__init__(self)
        self.state_queue = state_queue
        self.dist_queue = dist_queue
        self.name = name

    def run(self):
        with tf.Session() as session:
            root = path.join(path.dirname(__file__), "model", "policy", self.name)
            saver = tf.train.import_meta_graph(path.join(root, self.name + ".meta"), clear_devices=True)
            checkpoint = tf.train.latest_checkpoint(root)
            saver.restore(session, checkpoint)
            print(checkpoint)
            while True:
                state = self.state_queue.get()
                if state is None:
                    return
                board = np.ndarray(shape=(1, 15, 15, 5), dtype=np.float32)
                board[:, :, :, 0] = (state.board > 0)
                board[:, :, :, 1] = (state.board < 0)
                board[:, :, :, 2] = (state.board == 0)
                board[:, :, :, 3] = 0
                board[:, :, :, 4] = 1
                y = session.run("prob:0", feed_dict={
                    "y:0": np.zeros(shape=(1, 225)), "f:0": np.zeros(shape=1), "x:0": board
                }).reshape((15, 15))
                self.dist_queue.put(np.exp(y))


def run():
    states = [None, mp.Queue(), mp.Queue()]
    distributions = mp.Queue()
    players = [
        Agent(states[1], distributions, "alphanet-5-black"),
        Agent(states[-1], distributions, "alphanet-5-white"),
    ]
    for player in players:
        player.start()
    root = tk.Tk()
    root.wm_title("Alpha Gomoku")
    root.attributes("-topmost", True)
    app = Application(states, distributions, root)
    app.mainloop()
    states[1].put(None)
    states[-1].put(None)


run()
