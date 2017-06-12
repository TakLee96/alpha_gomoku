import pickle
from time import time
from sys import stdout
from state import State
from itertools import count
from os import path, listdir
from minimax import MinimaxAgent


latest = -1
for f in listdir(path.join(path.dirname(__file__), "data", "minimax")):
    if f.endswith(".pkl"):
        latest = max(latest, int(f.split(".")[0]))


agent = MinimaxAgent(max_depth=6, max_width=8)
names = ["draw", "black", "white"]
for i in count(latest + 1):
    print("[INFO] game %d begin" % i)
    begin = time()
    state = State()
    while len(state.history) != 225 and not state.end:
        x, y = agent.get_action(state)
        state.move(x, y)
        stdout.write(".")
        stdout.flush()
    winner = state.player if state.end else 0
    with open(path.join(path.dirname(__file__), "data", "minimax", "%d.pkl" % i), "wb") as out:
        pickle.dump({"history": state.history, "winner": winner }, out)
    print("\n[INFO] the winner is %s [%d seconds] [game stored]" % (names[winner], time() - begin))
