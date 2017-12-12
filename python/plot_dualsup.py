from matplotlib import pyplot as plt
import numpy as np

x = []
p = []
v = []

with open("dual_fail/dualsup.log", "r") as file:
    s = file.read()
    sections = s.split("\nstep")
    for i in range(1, len(sections)):
        section = sections[i]
        iteration = int(section.split("analysis")[0])
        policy_loss = float(section.split("\n")[1].split("policy loss:")[1])
        value_loss = float(section.split("\n")[2].split("value loss:")[1])
        x.append(iteration)
        p.append(policy_loss)
        v.append(value_loss)

x = np.array(x)
p = np.array(p)
v = np.array(v)

plt.title("Loss vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(x, p, 'r-', label="Cross-Entropy Loss")
plt.plot(x, v/4, 'b-', label="Regression Loss")
plt.legend()
plt.show()
