from matplotlib import pyplot as plt
import numpy as np

x = []
p = []
v = []

with open("treesup.log", "r") as file:
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

plt.title("Loss vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.plot(x, p, 'r-', label="Cross-Entropy Loss")
plt.plot(x, v, 'b-', label="Regression Loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
