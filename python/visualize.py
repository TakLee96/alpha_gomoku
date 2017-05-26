""" visualize games from dataset """
import tkinter as tk
import codecs
from os import path
from sys import argv
from state import State


class Application(tk.Frame):
    def __init__(self, history, master):
        tk.Frame.__init__(self, master)
        self.button = list()
        self.frames = list()
        self.state = State()
        self.history = history
        self.index = 0
        root = path.join(path.dirname(__file__), "img")
        self.image = [
          tk.PhotoImage(file=path.join(root, "empty.gif")),
          tk.PhotoImage(file=path.join(root, "naught.gif")),
          tk.PhotoImage(file=path.join(root, "cross.gif")),
        ]
        self.pack()
        self.create_widgets()

    def highlight(self, x, y):
        for i, j in self.state.highlight(x, y):
            self.frames[i*15+j].config(padx=1, pady=1, bg="green")

    def click(self, e):
        if self.index < len(self.history):
            x, y = self.history[self.index]
            self.button[x*15+y].config(image=self.image[self.state.player])
            self.state.move(x, y)
            self.index += 1
            if self.state.end:
                self.highlight(x, y)

    def create_widgets(self):
        for i in range(15):
            for j in range(15):
                f = tk.Frame(self, height=50, width=50)
                f.pack_propagate(0)
                f.grid(row=i, column=j, padx=0, pady=0)
                self.frames.append(f)
                b = tk.Label(f, image=self.image[0], bg="red")
                b.pack(fill=tk.BOTH, expand=1)
                b.bind("<Button-1>", self.click)
                self.button.append(b)

def convert(char):
  if not (97 <= ord(char) <= 111):
    print(char)
    raise Exception("damn it")
  return ord(char) - 97

def get_move(string):
  if len(string) != 5:
    print(string)
    raise Exception("fuck you")
  return (convert(string[2]), convert(string[3]))

if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python visualize.py [game number]")
    else:
        directory = path.join(path.dirname(__file__), "data", "raw")
        with codecs.open(path.join(directory, "%.04d.sgf" % int(argv[1])), "r", encoding='utf-8', errors='ignore') as file:
            string = file.read().strip()
            index = string.find(";B[")
            moves = list(map(get_move, string[(index+1):-2].split(";")))
        root = tk.Tk()
        root.wm_title("Game " + argv[1])
        root.attributes("-topmost", True)
        app = Application(moves, root)
        app.mainloop()
