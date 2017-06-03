""" battle between agent and human """
import numpy as np
import tkinter as tk
import tensorflow as tf
import multiprocessing as mp
from os import path
from state import State
from feature import diff


class Application(tk.Frame):
    def __init__(self, state_queue, value_queue, master):
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
        self.value_queue = value_queue
        self.pack()
        self.create_widgets()
        self.draw_probability()

    def draw_probability(self):
        for i in range(15):
            for j in range(15):
                if self.state.board[i, j] == 0:
                    new = diff(self.state, i, j)
                    if new["-o-oo-"] + new["-ooo-"] >= 2 or \
                        new["four-o"] + new["-oooo-"] >= 2 or self.state._long(i, j):
                        self.button[i * 15 + j].config(image="", text="%.2f" % -1)
                    else:
                        self.state.move(i, j)
                        self.state_queue[self.state.player].put(self.state)
                        self.button[i * 15 + j].config(image="", text="%.2f" % self.value_queue.get())
                        self.state.rewind()
        # self.state_queue[self.state.player].put(self.state)
        # print(self.value_queue.get()[0, 0])

    def highlight(self, x, y):
        for i, j in self.state.highlight(x, y):
            self.frames[i*15+j].config(padx=1, pady=1, bg="blue")

    def click(self, i, j):
        def respond(e):
            if not self.state.end and self.state.board[i, j] == 0:
                self.button[i*15+j].config(image=self.image[self.state.player])
                self.state.move(i, j)
                if self.state.end:
                    if self.state._win(i, j):
                        self.highlight(i, j)
                    else:
                        self.frames[i*15+j].config(padx=1, pady=1, bg="red")
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
                b = tk.Label(f, image=self.image[0], bg="yellow")
                b.pack(fill=tk.BOTH, expand=1)
                b.bind("<Button-1>", self.click(i, j))
                self.button.append(b)


class Agent(mp.Process):
    def __init__(self, state_queue, value_queue, name):
        mp.Process.__init__(self)
        self.state_queue = state_queue
        self.value_queue = value_queue
        self.name = name

    def run(self):
        with tf.Session() as session:
            root = path.join(path.dirname(__file__), "model", "value", self.name)
            saver = tf.train.import_meta_graph(path.join(root, self.name + ".meta"), clear_devices=True)
            saver.restore(session, tf.train.latest_checkpoint(root))
            while True:
                state = self.state_queue.get()
                if state is None:
                    return
                y = session.run("y:0", feed_dict={
                    "x:0": state.board.reshape((1, 225)),
                    "y_:0": np.zeros(shape=(1, 1))})
                self.value_queue.put(y)


def run():
    states = [None, mp.Queue(), mp.Queue()]
    values = mp.Queue()
    players = [Agent(states[1], values, "qbtnet-new-black"), Agent(states[-1], values, "qbtnet-new-white")]
    for player in players:
        player.start()
    root = tk.Tk()
    root.wm_title("Alpha Gomoku")
    root.attributes("-topmost", True)
    app = Application(states, values, root)
    app.mainloop()
    states[1].put(None)
    states[-1].put(None)


run()
