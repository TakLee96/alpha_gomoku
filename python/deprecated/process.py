""" data preprocessing """
from state import State
from feature import diff
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


def minimax():
    root = path.join(path.dirname(__file__), "data", "minimax")
    black_boards = list()
    black_actions = list()
    black_feedbacks = list()
    white_boards = list()
    white_actions = list()
    white_feedbacks = list()
    
    for f in listdir(root):
        if f.endswith(".pkl"):
            with open(path.join(root, f), "rb") as file:
                d = pickle.load(file)
                moves = d["history"]
                winner = d["winner"]
                if winner != 0:
                    for change in changes:
                        new_moves = list(map(change, moves))
                        state = State()
                        for i, t in enumerate(new_moves):
                            board = np.ndarray(shape=(15, 15, 5), dtype=bool)
                            board[:, :, 0] = (state.board > 0)
                            board[:, :, 1] = (state.board < 0)
                            board[:, :, 2] = (state.board == 0)
                            board[:, :, 3] = 0
                            board[:, :, 4] = 1
                            if i >= 2 or change((0, 1)) == (0, 1):
                                if i % 2 == 0:
                                    if winner == 1:
                                        black_boards.append(board)
                                        black_actions.append(np.ravel_multi_index(t, dims=(15, 15)))
                                        black_feedbacks.append(1)
                                    elif i < len(moves) - 6:
                                        black_boards.append(board)
                                        black_actions.append(np.ravel_multi_index(t, dims=(15, 15)))
                                        black_feedbacks.append(-1)
                                else:
                                    if winner == -1:
                                        white_boards.append(board)
                                        white_actions.append(np.ravel_multi_index(t, dims=(15, 15)))
                                        white_feedbacks.append(1)
                                    elif i < len(moves) - 6:
                                        white_boards.append(board)
                                        white_actions.append(np.ravel_multi_index(t, dims=(15, 15)))
                                        white_feedbacks.append(-1)                         
                            state.move(*t)
                print("processed %s" % f)

    X_b = np.array(black_boards, dtype=np.uint8)
    y_b = np.array(black_actions, dtype=np.uint8)
    f_b = np.array(black_feedbacks, dtype=np.float32)
    X_w = np.array(white_boards, dtype=np.uint8)
    y_w = np.array(white_actions, dtype=np.uint8)
    f_w = np.array(white_feedbacks, dtype=np.float32)

    n_b, _, _, _ = X_b.shape
    n_w, _, _, _ = X_w.shape

    ordering_b = np.random.permutation(n_b)
    ordering_w = np.random.permutation(n_w)

    X_b = X_b[ordering_b, :, :, :]
    y_b = y_b[ordering_b]
    f_b = f_b[ordering_b]
    X_w = X_w[ordering_w, :, :, :]
    y_w = y_w[ordering_w]
    f_w = f_w[ordering_w]

    m_b = n_b // 10
    m_w = n_w // 10

    X_b_t = X_b[m_b:, :, :, :]
    y_b_t = y_b[m_b:]
    f_b_t = f_b[m_b:]
    X_w_t = X_w[m_w:, :, :, :]
    y_w_t = y_w[m_w:]
    f_w_t = f_w[m_w:]
    X_b_v = X_b[:m_b, :, :, :]
    y_b_v = y_b[:m_b]
    f_b_v = f_b[:m_b]
    X_w_v = X_w[:m_w, :, :, :]
    y_w_v = y_w[:m_w]
    f_w_v = f_w[:m_w]

    m_b = { "X_v": X_b_v, "y_v": y_b_v, "f_v": f_b_v, "X_t": X_b_t, "y_t": y_b_t, "f_t": f_b_t }
    m_w = { "X_v": X_w_v, "y_v": y_w_v, "f_v": f_w_v, "X_t": X_w_t, "y_t": y_w_t, "f_t": f_w_t }
    savemat(path.join(path.dirname(__file__), "data", "minimax_black"), m_b, do_compression=True)
    savemat(path.join(path.dirname(__file__), "data", "minimax_white"), m_w, do_compression=True)


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

def svm():
    root = path.join(path.dirname(__file__), "data", "minimax")
    score = dict()
    for f in listdir(root):
        if f.endswith(".pkl"):
            with open(path.join(root, f), "rb") as file:
                d = pickle.load(file)
                moves = d["history"]
                winner = d["winner"]
                state = State()
                for x, y in moves:
                    feature = tuple(translate(state.features, state.player))
                    if feature in score:
                        n, avg = score[feature]
                        score[feature] = (n + 1, (n * avg + winner) / (n + 1))
                    else:
                        score[feature] = (1, winner * 1.0)
                    state.move(x, y)
                feature = tuple(translate(state.features, 1 if len(state.history) % 2 == 0 else -1))
                if feature in score:
                    n, avg = score[feature]
                    score[feature] = (n + 1, (n * avg + winner) / (n + 1))
                else:
                    score[feature] = (1, winner * 1.0)
                print("%s done" % f)
    n = len(score)
    X = np.zeros(shape=(n, 29), dtype=np.uint8)
    y = np.zeros(shape=n, dtype=np.float64)
    for i, (k, v) in enumerate(score.items()):
        X[i, :] = list(k)
        y[i] = v[1]

    ordering = np.random.permutation(n)
    X = X[ordering, :]
    y = y[ordering]

    m = n // 8
    X_v = X[:m, :]
    y_v = y[:m]
    X_t = X[m:, :]
    y_t = y[m:]

    mat = { "X_v": X_v, "y_v": y_v, "X_t": X_t, "y_t": y_t }
    savemat(path.join(path.dirname(__file__), "data", "svm-avg"), mat, do_compression=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("raw", "minimax", "dual", "svm"):
        print("Usage: python process.py [raw/minimax/dual/svm]")
    else:
        if sys.argv[1] == "raw":
            raw()
        elif sys.argv[1] == "minimax":
            minimax()
        elif sys.argv[1] == "svm":
            svm()
        else:
            dual()
