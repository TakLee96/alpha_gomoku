""" battle between agent and human """
import numpy as np
import tkinter as tk
from os import path
from time import time
from state import State
from feature import diff
from minimax import MinimaxAgent


class Application(tk.Frame):
    def __init__(self, master):
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
        self.agent = MinimaxAgent()
        self.last = None
        self.pack()
        self.create_widgets()
        self.recommend()

    def recommend(self):
        t = time()
        actions = self.agent.get_score(self.state)
        print("time elapsed: %f seconds" % (time() - t))
        if self.last is not None:
            for x, y, _ in self.last:
                button = self.button[np.ravel_multi_index((x, y), dims=(15, 15))]
                button.config(image=self.image[self.state.board[x, y]])
        for x, y, v in actions:
            button = self.button[np.ravel_multi_index((x, y), dims=(15, 15))]
            button.config(image="", text="%.02f" % v)
        self.last = actions

    def highlight(self, x, y):
        for i, j in self.state.highlight(x, y):
            self.frames[np.ravel_multi_index((i, j), dims=(15, 15))].config(padx=1, pady=1, bg="blue")

    def click(self, i, j):
        def respond(e):
            if not self.state.end and self.state.board[i, j] == 0:
                self.button[np.ravel_multi_index((i, j), dims=(15, 15))].config(image=self.image[self.state.player])
                self.state.move(i, j)
                if self.state.end:
                    if self.state._win(i, j):
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
app = Application(root)
app.mainloop()
