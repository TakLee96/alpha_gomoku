from matplotlib import pyplot as plt
import numpy as np

"""
Rank Name              Elo    +    - games score oppo. draws 
   1 supervised-4000   940  187  113   800  100%  -118    0% 
   2 dagger-140        164   28   27   800   68%   -20    0% 
   3 dagger-126        132   27   26   800   66%   -16    0% 
   4 dagger-107        -12   25   25   800   53%     1    0% 
   5 dagger-60         -36   26   25   800   51%     4    0% 
   6 dagger-84         -73   25   25   800   47%     9    0% 
   7 dagger-40        -149   25   25   800   40%    19    0% 
   8 dagger-21        -442   31   33   800   15%    55    0% 
   9 dagger-0         -523   35   37   800   10%    65    0% 
"""

x = np.array([    0,   21,   40,  60,  84, 107, 126, 140 ])
e = np.array([ -523, -442, -149, -36, -73, -12, 132, 164 ])

plt.title("Elo Ratings vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Elo Ratings")
plt.plot(x, e, 'r-', label="Dagger")
plt.plot(x, 940 * np.ones(shape=x.shape), 'b-', label="Supervised")
plt.legend()
plt.show()
