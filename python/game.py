""" Game GUI """
import Tkinter as tk
import numpy as np
import tensorflow as tf
from gomoku import State

class Application(tk.Frame):
  def __init__(self, session, master):
    tk.Frame.__init__(self, master)
    self.button = list()
    self.state = State()
    self.image = [
      tk.PhotoImage(file="img/empty.gif"),
      tk.PhotoImage(file="img/naught.gif"),
      tk.PhotoImage(file="img/yellow.gif"),
      tk.PhotoImage(file="img/red.gif"),
      tk.PhotoImage(file="img/cross.gif"),
    ]
    self.pack()
    self.create_widgets()
    self.recommended_moves = self.recommend_moves()
    for x, y in self.recommended_moves:
      self.button[x*15+y].config(image=self.image[3])

  def recommend_moves(self):
    prob = session.run("Reshape_1:0", feed_dict={
      "x:0": (self.state.player * self.state.board).reshape((1, 225)),
      "y_:0": np.zeros(shape=(1, 225)),
      "is_training:0": False}).reshape((15, 15))
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
  saver = tf.train.import_meta_graph("model/sdknet/sdknet.meta", clear_devices=True)
  saver.restore(session, "model/sdknet/sdknet-25000")
  root = tk.Tk()
  app = Application(session, root)
  app.mainloop()
