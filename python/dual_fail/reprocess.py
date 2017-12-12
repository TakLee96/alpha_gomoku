import numpy as np
from state import State
from scipy.io import loadmat, savemat


wikipedia = [
    "-xxo", "-x-xo", "-oox", "-o-ox",
    "-x-x-", "-xx-", "-o-o-", "-oo-",
    "-x-xxo", "-xxxo", "-o-oox", "-ooox",
    "-x-xx-", "-xxx-", "-o-oo-", "-ooo-",
    "-xxxx-", "-xxxxo", "-oooo-", "-oooox",
    "four-o", "four-x", "win-o", "win-x", "violate"
]
wikipedia = { k: i for i, k in enumerate(wikipedia) }


m = loadmat("value_treesup_new")
X = m["X"]
V = m["V"][0]
nX = []
for i in range(3000):
    black = X[i,:,:,0].reshape(225)
    white = X[i,:,:,1].reshape(225)
    empty = X[i,:,:,2].reshape(225)
    assert np.all((black+white+empty)==1), "broken"
    black_moves = [e[0] for e in np.argwhere(black)]
    white_moves = [e[0] for e in np.argwhere(white)]
    assert len(black_moves) == len(white_moves) or len(black_moves) == len(white_moves) + 1, "fuck"
    s = State()
    for j in range(len(black_moves) + len(white_moves)):
        if j % 2 == 0:
            s.feature_move(*np.unravel_index(black_moves[j // 2], dims=(15, 15)))
        else:
            s.feature_move(*np.unravel_index(white_moves[j // 2], dims=(15, 15)))
    built = s.featurize()
    assert np.all(built == X[i])
    features = np.zeros(shape=2*len(wikipedia), dtype=np.float32)
    for feature in s.features.keys():
        if s.features[feature] > 0:
            assert feature in wikipedia, "unknown " + feature
            location = wikipedia[feature]
            if s.player == 1:
                location = location
            else:
                location = location + len(wikipedia)
            features[location] = s.features[feature]
    nX.append(features)
savemat("fc_value", {"X": nX, "V": V}, do_compression=True)