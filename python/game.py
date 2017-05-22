""" Game GUI """
import tkinter as tk
import numpy as np
import tensorflow as tf
from os import path
from gomoku import State


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
    def __init__(self, session, master):
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
        self.session = session
        self.pack()
        self.create_widgets()
        self.recommended_moves = self.recommend_moves()
        self.draw_probability()

    def recommend_moves(self):
        mean = np.zeros(shape=225, dtype=float)
        for i in range(8):
            prob = self.session.run(tensor_name, feed_dict={
                "x:0": changes[i](self.state.player * self.state.board).reshape((1, 225)),
                "y_:0": np.zeros(shape=(1, 225))}).reshape((15, 15))
            prob = reverses[i](np.exp(prob - prob.max())) * (self.state.board == 0)
            prob = prob / prob.sum()
            mean += prob.reshape(225)
        mean = mean / mean.sum()
        moves = mean.argsort()[::-1]
        result = list()
        while mean[moves[len(result)]] > 0.01:
            x, y = np.unravel_index(moves[len(result)], (15, 15))
            result.append((x, y, mean[moves[len(result)]]))
        return result

    def draw_probability(self):
        max_prob = self.recommended_moves[0][2]
        for x, y, p in self.recommended_moves:
            if np.isclose(p, max_prob):
                color = "red"
            else:
                color = "yellow"
            self.button[x * 15 + y].config(image="", text="%.2f" % p, bg=color)

    def highlight(self, x, y):
        for i, j in self.state.highlight(x, y):
            self.frames[i*15+j].config(padx=1, pady=1, bg="green")

    def click(self, i, j):
        def respond(e):
            if not self.state.end and self.state.board[i, j] == 0:
                self.button[i*15+j].config(image=self.image[self.state.player])
                self.state.move(i, j)
                for x, y, _ in self.recommended_moves:
                    self.button[x*15+y].config(image=self.image[self.state.board[x, y]])
                if self.state.end:
                    self.highlight(i, j)
                else:
                    self.recommended_moves = self.recommend_moves()
                    self.draw_probability()
        return respond

    def create_widgets(self):
        for i in range(15):
            for j in range(15):
                f = tk.Frame(self, height=50, width=50)
                f.pack_propagate(0)
                f.grid(row=i, column=j, padx=0, pady=0)
                self.frames.append(f)
                b = tk.Label(f, image=self.image[0], bg="red")
                b.pack(fill=tk.BOTH, expand=1)
                b.bind("<Button-1>", self.click(i, j))
                self.button.append(b)


def run():
    with tf.Session() as session:
        name = "deepsdknet"
        checkpoint = 30000
        root = path.join(path.dirname(__file__), "model", name)
        saver = tf.train.import_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
        saver.restore(session, path.join(root, name + "-" + str(checkpoint)))
        root = tk.Tk()
        root.wm_title("Alpha Gomoku")
        root.focus_force()
        app = Application(session, root)
        app.mainloop()


tensor_name = "y:0"  # "Reshape_1:0" for sdknet
run()
