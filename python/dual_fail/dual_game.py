""" battle between agent and human """

import argparse
import numpy as np
import tkinter as tk
import tensorflow as tf
from os import path
from time import time
from state import State
from feature import diff
from dual_agent import DualAgent
from mcts_agent import MCTSAgent

class Application(tk.Frame):
    def __init__(self, agent, master, ensemble):
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
        self.agent = agent
        self.ensemble = ensemble
        self.pack()
        self.create_widgets()
        self.recommend()

    def recommend(self):
        t = time()
        if not self.ensemble:
            dist = self.agent.get_dist(self.state).reshape((15, 15))
        else:
            dist = self.agent.get_dist_ensemble(self.state).reshape((15, 15))
        print("time elapsed: %f seconds" % (time() - t))
        for x in range(15):
            for y in range(15):
                button = self.button[np.ravel_multi_index((x, y), dims=(15, 15))]
                if dist[x, y] > 0:
                    button.config(image="", text="%.02f" % dist[x, y])
                else:
                    button.config(image=self.image[self.state.board[x, y]])

    def highlight(self, x, y):
        for i, j in self.state.highlight(x, y):
            self.frames[np.ravel_multi_index((i, j), dims=(15, 15))].config(padx=1, pady=1, bg="blue")

    def click(self, i, j):
        def respond(e):
            if not self.state.end and self.state.board[i, j] == 0:
                self.button[np.ravel_multi_index((i, j), dims=(15, 15))].config(image=self.image[self.state.player])
                self.state.move(i, j)
                if hasattr(self.agent, "update"):
                    # self.agent.refresh()
                    self.agent.update(self.state)
                if self.state.end:
                    if self.state.features["win-o"] + self.state.features["win-x"] > 0:
                        self.highlight(i, j)
                    else:
                        self.frames[np.ravel_multi_index((i, j), dims=(15, 15))].config(padx=1, pady=1, bg="red")
                else:
                    self.recommend()
        return respond

    def create_widgets(self):
        for i in range(15):
            for j in range(15):
                f = tk.Frame(self, height=50, width=50)
                f.pack_propagate(0)
                f.grid(row=i, column=j, padx=0, pady=0)
                self.frames.append(f)
                b = tk.Label(f, image=self.image[0], bg="yellow")
                b.pack(fill=tk.BOTH, expand=1)
                b.bind("<Button-1>", self.click(i, j))
                self.button.append(b)

root = tk.Tk()
root.wm_title("Alpha Gomoku")
root.attributes("-topmost", True)

with tf.Session() as sess:
    agent = MCTSAgent(sess, "treesup")
    # agent = DualAgent(sess, "dualdagger")
    app = Application(agent, root, ensemble=False)
    app.mainloop()
