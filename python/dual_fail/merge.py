import numpy as np
from scipy.io import loadmat, savemat


m = loadmat("value_treesup_mat/value_treesup_1")
X = m["X"]
V = m["V"][0]
for i in range(2, 40):
    m = loadmat("value_treesup_mat/value_treesup_%d" % i)
    X = np.append(X, m["X"], axis=0)
    V = np.append(V, m["V"][0], axis=0)

savemat("value_treesup_merged", {"X": X, "V": V}, do_compression=True)
