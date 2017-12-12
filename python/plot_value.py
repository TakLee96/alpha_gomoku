from matplotlib import pyplot as plt
import numpy as np

x = []
trn = []
val = []
reg = []

with open("dual_fail/value-192.log", "r") as file:
    s = file.read()
    sections = s.split("\n")
    for section in sections:
        if section.startswith("step"):
            tokens = section.split(" ")
            x.append(int(tokens[1]))
            trn.append(float(tokens[5]))
            val.append(float(tokens[7]))
            reg.append(float(tokens[9]))

x = np.array(x)
trn = np.array(trn)
val = np.array(val)
reg = np.array(reg)

plt.title("Loss vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(x, trn, 'r-', label="Training Loss")
plt.plot(x, val, 'b-', label="Validation Loss")
plt.plot(x, reg, 'g-', label="Regularization Loss")
plt.legend()
plt.show()
