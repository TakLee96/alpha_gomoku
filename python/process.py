""" data preprocessing """
from state import State
from os import listdir, path
from scipy.io import savemat
import sys
import codecs
import pickle
import numpy as np


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


def raw():
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


def godsdknet():
    root = path.join(path.dirname(__file__), "data", "godsdknet")
    black_boards = list()
    black_scores = list()
    white_boards = list()
    white_scores = list()
    for f in listdir(root):
        with open(path.join(root, f), "rb") as file:
            d = pickle.load(file)
            moves = d["history"]
            winner = d["winner"]
            state = State()
            black_boards.append(np.copy(state.board.reshape(225)))
            black_scores.append(0)
            for i, (x, y) in enumerate(moves):
                state.move(x, y)
                if i % 2 == 0:
                    white_boards.append(np.copy(state.board.reshape(225)))
                    white_scores.append(winner)
                else:
                    black_boards.append(np.copy(state.board.reshape(225)))
                    black_scores.append(winner)
    black_boards = np.array(black_boards)
    black_scores = np.array(black_scores)
    white_boards = np.array(white_boards)
    white_scores = np.array(white_scores)
    n_black = len(black_scores)
    n_white = len(white_scores)
    black_ordering = np.random.permutation(n_black)
    white_ordering = np.random.permutation(n_white)
    black_boards = black_boards[black_ordering, :]
    black_scores = black_scores[black_ordering]
    white_boards = white_boards[white_ordering, :]
    white_scores = white_scores[white_ordering]
    X_b_v = black_boards[:10000, :]
    y_b_v = black_scores[:10000]
    X_b_t = black_boards[10000:, :]
    y_b_t = black_scores[10000:]
    X_w_v = white_boards[:10000, :]
    y_w_v = white_scores[:10000]
    X_w_t = white_boards[10000:, :]
    y_w_t = white_scores[10000:]
    m_b = { "X_v": X_b_v, "y_v": y_b_v, "X_t": X_b_t, "y_t": y_b_t }
    m_w = { "X_v": X_w_v, "y_v": y_w_v, "X_t": X_w_t, "y_t": y_w_t }
    savemat(path.join(path.dirname(__file__), "data", "black"), m_b, do_compression=True)
    savemat(path.join(path.dirname(__file__), "data", "white"), m_w, do_compression=True)


def dual():
    root = path.join(path.dirname(__file__), "data", "raw")
    black_boards = list()
    black_actions = list()
    white_boards = list()
    white_actions = list()
    
    for f in listdir(root):
        with codecs.open(path.join(root, f), "r", encoding="utf-8", errors="ignore") as file:
            string = file.read().strip()
            index = string.find(";B[")
            moves = list(map(get_move, string[(index+1):-2].split(";")))
            try:
                valid = True
                state = State()
                for t in moves:
                    state.move(*t)
            except:
                valid = False
                print("file %s violates renju" % f)
            if valid:
                for change in changes:
                    new_moves = list(map(change, moves))
                    state = State()
                    for i, t in enumerate(new_moves):
                        if i % 2 == 0:
                            black_boards.append(np.copy(state.board.reshape(225)))
                            black_actions.append(np.ravel_multi_index(t, dims=(15, 15)))
                            state.move(*t)
                        else:            
                            white_boards.append(np.copy(state.board.reshape(225)))
                            white_actions.append(np.ravel_multi_index(t, dims=(15, 15)))
                            state.move(*t)    
    X_b = np.array(black_boards, dtype=np.int8)
    y_b = np.array(black_actions, dtype=np.uint8)
    X_w = np.array(white_boards, dtype=np.int8)
    y_w = np.array(white_actions, dtype=np.uint8)

    n_b, d = X_b.shape
    n_w, d = X_w.shape

    ordering_b = np.random.permutation(n_b)
    ordering_w = np.random.permutation(n_w)

    X_b = X_b[ordering_b, :]
    y_b = y_b[ordering_b]
    X_w = X_w[ordering_w, :]
    y_w = y_w[ordering_w]

    m_b = n_b // 10
    m_w = n_w // 10

    X_b_t = X_b[m_b:, :]
    y_b_t = y_b[m_b:]
    X_w_t = X_w[m_w:, :]
    y_w_t = y_w[m_w:]
    X_b_v = X_b[:m_b, :]
    y_b_v = y_b[:m_b]
    X_w_v = X_w[:m_w, :]
    y_w_v = y_w[:m_w]

    m_b = { "X_v": X_b_v, "y_v": y_b_v, "X_t": X_b_t, "y_t": y_b_t }
    m_w = { "X_v": X_w_v, "y_v": y_w_v, "X_t": X_w_t, "y_t": y_w_t }
    savemat(path.join(path.dirname(__file__), "data", "policy_black"), m_b, do_compression=True)
    savemat(path.join(path.dirname(__file__), "data", "policy_white"), m_w, do_compression=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("raw", "godsdknet", "dual"):
        print("Usage: python process.py [raw/godsdknet/dual]")
    else:
        if sys.argv[1] == "raw":
            raw()
        elif sys.argv[1] == "godsdknet":
            godsdknet()
        else:
            dual()
