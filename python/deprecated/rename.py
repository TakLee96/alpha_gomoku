from pickle import load
from os import listdir, path, rename


# root = path.join(path.dirname(__file__), "data", "minimax")
# for i, f in enumerate(listdir(root)):
#     if f.endswith(".pkl"):
#         rename(path.join(root, f), path.join(root, "m-%.05d.pkl" % i))
# for i, f in enumerate(listdir(root)):
#     if f.endswith(".pkl"):
#         rename(path.join(root, f), path.join(root, "%.05d.pkl" % i))


# everything = []
# root = path.join(path.dirname(__file__), "data", "minimax")
# for f in listdir(root):
#     if f.endswith(".pkl"):
#         with open(path.join(root, f), "rb") as file:
#             moves = load(file)["history"]
#             for name, old_moves in everything:
#                 if moves == old_moves:
#                     print("%s is the same with %s" % (f, name))
#             everything.append((f, moves))
#             print("... done comparing for %s %d" % (f, len(moves)))


# count = [0, 0, 0]
# root = path.join(path.dirname(__file__), "data", "minimax")
# for f in listdir(root):
#     if f.endswith(".pkl"):
#         with open(path.join(root, f), "rb") as file:
#             winner = load(file)["winner"]
#             count[winner] += 1
# print("black: %d; white: %d" % (count[1], count[-1]))
