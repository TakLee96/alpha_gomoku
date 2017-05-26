""" data preprocessing """
from os import listdir, path
from scipy.io import savemat
import numpy as np


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

changes = [
    lambda t: (     t[0],      t[1]),    # original
    lambda t: (     t[1], 14 - t[0]),    # rotate 90
    lambda t: (14 - t[0], 14 - t[1]),    # rotate 180
    lambda t: (14 - t[1],      t[0]),    # rotate 270
    lambda t: (     t[0], 14 - t[1]),    # y-flip
    lambda t: (     t[1],      t[0]),    # rotate 90 + y-flip
    lambda t: (14 - t[0],      t[1]),    # rotate 180 + y-flip
    lambda t: (14 - t[1], 14 - t[0]),    # rotate 270 + y-flip
]

root = path.join(path.dirname(__file__), "data", "raw")
boards = list()
actions = list()
for f in listdir(root):
    with open(path.join(root, f)) as file:
        string = file.read().strip()
        index = string.find(";B[")
        moves = map(get_move, string[(index+1):-2].split(";"))
        for change in changes:
            new_moves = map(change, moves)
            board = np.zeros(shape=(15, 15), dtype=np.int8)
            for i, t in enumerate(new_moves):
                x, y = t
                assert 0 <= x <= 14 and 0 <= y <= 14
                boards.append(np.copy(board.reshape(225)))
                actions.append(t[0] * 15 + t[1])
                board[t] = (i % 2 == 0) * 2 - 1

X = np.array(boards)
y = np.array(actions)
n, d = X.shape
ordering = np.random.permutation(n)
X = X[ordering, :]
y = y[ordering]

m = n // 10
X_v = X[:m, :]
y_v = y[:m]
X_t = X[m:, :]
y_t = y[m:]

m = { "X_v": X_v, "y_v": y_v, "X_t": X_t, "y_t": y_t }
savemat(path.join(path.dirname(__file__), "data", "formatted"), m, do_compression=True)
