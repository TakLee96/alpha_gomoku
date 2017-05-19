""" Game GUI """
import Tkinter as tk
import numpy as np
import tensorflow as tf
from os import path
from gomoku import State


class Application(tk.Frame):
  def __init__(self, session, master):
    tk.Frame.__init__(self, master)
    self.button = list()
    self.state = State()
    root = path.join(path.dirname(__file__), "img")
    self.image = [
      tk.PhotoImage(file=path.join(root, "empty.gif")),
      tk.PhotoImage(file=path.join(root, "naught.gif")),
      tk.PhotoImage(file=path.join(root, "yellow.gif")),
      tk.PhotoImage(file=path.join(root, "red.gif")),
      tk.PhotoImage(file=path.join(root, "corss.gif")),
    ]
    self.pack()
    self.create_widgets()
    self.recommended_moves = self.recommend_moves()
    for x, y in self.recommended_moves:
      self.button[x*15+y].config(image=self.image[3])

  def recommend_moves(self):
    # Reshape_1:0
    prob = session.run("y", feed_dict={
      "x:0": (self.state.player * self.state.board).reshape((1, 225)),
      "y_:0": np.zeros(shape=(1, 225)),
      "is_training:0": False})
    maxv = prob.max()
    moves = list()
    location = (prob == maxv)
    for i in range(15):
      for j in range(15):
        if location[i, j]:
          moves.append((i, j))
    return moves

  def interpret(self, reshaped):
    loc = reshaped.argmax()
    x = loc / 15
    y = loc % 15
    return x, y

  def highlight(self, x, y):
    for i, j in self.state.highlight(x, y):
      self.button[i*15+j].config(image=self.image[2])

  def click(self, i, j):
    def respond(e):
      if not self.state.end and self.state.board[i, j] == 0:
        self.button[i*15+j].config(image=self.image[self.state.player])
        self.state.move(i, j)
        for x, y in self.recommended_moves:
          self.button[x*15+y].config(image=self.image[self.state.board[x, y]])
        if self.state.end:
          self.highlight(i, j)
        else:
          self.recommended_moves = self.recommend_moves()
          for x, y in self.recommended_moves:
            self.button[x * 15 + y].config(image=self.image[3])
    return respond

  def create_widgets(self):
    for i in range(15):
      for j in range(15):
        b = tk.Label(self, image=self.image[0], width="45", height="45")
        b.grid(row=i, column=j, padx=0, pady=0)
        b.bind("<Button-1>", self.click(i, j))
        self.button.append(b)


with tf.Session() as session:
  name = "sdknet"
  checkpoint = 25000
  root = path.join(path.dirname(__file__), "model", name)
  saver = tf.train.import_meta_graph(path.join(root, name + ".meta"), clear_devices=True)
  saver.restore(session, path.join(root, name + "-" + str(checkpoint)))
  root = tk.Tk()
  root.wm_title("Alpha Gomoku")
  root.focus_force()
  app = Application(session, root)
  app.mainloop()
