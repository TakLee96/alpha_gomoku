import numpy as np
import tkinter as tk
from os import path
from sys import argv
from state import State
from sklearn.externals import joblib


fundamental = {
    "-xxo": 0, "-x-xo": 1, "-oox": 2, "-o-ox": 3,
    "-x-x-": 4, "-xx-": 5, "-o-o-": 6, "-oo-": 7,
    "-x-xxo": 8, "-xxxo": 9, "-o-oox": 10, "-ooox": 11,
    "-x-xx-": 12, "-xxx-": 13, "-o-oo-": 14, "-ooo-": 15,
    "-xxxx-": 16, "-xxxxo": 17, "-oooo-": 18, "-oooox": 19,
    "win-o": 20, "win-x": 21, "four-o": 22, "four-x": 23,
    "o-o-o": 24, "x-x-x": 25, "violate": 26
}
def translate(features, player):
    feature = np.zeros(shape=29, dtype=int)
    if player == 1:
        feature[27] = 1
    else:
        feature[28] = 1
    for k, v in features.items():
        if v > 0:
            feature[fundamental[k]] = v
    return feature


class Application(tk.Frame):
    def __init__(self, master, clf):
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
        self.pack()
        self.create_widgets()
        self.clf = clf
        print(self.clf.predict(translate(self.state.features, self.state.player).reshape(1, 29))[0])

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
                    print(self.clf.predict(translate(self.state.features, self.state.player).reshape(1, 29))[0])
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


if __name__ == "__main__":
    if len(argv) != 4:
        print("Usage: python game_svm.py [dataset] [kernel] [degree]")
    else:
        clf = joblib.load(path.join(path.dirname(__file__), "model", "value", "svm",
            "%s-%s-%d.pkl" % (argv[1], argv[2], int(argv[3]))))
        root = tk.Tk()
        root.wm_title("Alpha Gomoku")
        root.attributes("-topmost", True)
        app = Application(root, clf)
        app.mainloop()
