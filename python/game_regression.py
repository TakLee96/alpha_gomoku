""" battle between agent and human """
import tkinter as tk
import numpy as np
import tensorflow as tf
from os import path, system
from state import State
from feature import diff


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
        self.draw_probability()

    def draw_probability(self):
        if self.state.player == 1:
            y = self.session.run("y:0", feed_dict={"x:0": self.state.board.reshape((1, 225)), "y_:0": np.zeros(shape=(1, 1))})
            print("winning probability for black: {0}%".format((y[0, 0] + 1) * 50))

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
                b = tk.Label(f, image=self.image[0], bg="red")
                b.pack(fill=tk.BOTH, expand=1)
                b.bind("<Button-1>", self.click(i, j))
                self.button.append(b)


def run():
    with tf.Session() as session:
        name = "qbtnet-black"
        checkpoint = 16000
        root = path.join(path.dirname(__file__), "model", name)
        saver = tf.train.import_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
        saver.restore(session, path.join(root, name + "-" + str(checkpoint)))
        root = tk.Tk()
        root.wm_title("Alpha Gomoku")
        root.attributes("-topmost", True)
        app = Application(session, root)
        app.mainloop()


run()
