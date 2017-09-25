""" policy gradient """

from agent import Agent
from state import State
import data_util as util

""" hyperparameters """
PG_ITERS = 5000
NUM_TRAJECTORY = 100
DISCOUNT = 0.9

""" begin training """
with tf.Session() as sess:
    for i in range(PG_ITERS):
        a = Agent(sess, "supervised", "supervised")

        X = []
        Y = []
        A = []
        for j in range(NUM_TRAJECTORY):
            s = State()
            while not s.end and len(s.history) < 225:
                s.move(a.get_action(s))
            if s.end:
                score = (DISCOUNT ** len(s.history)) * float(s.player)
                moves = s.history
                s = State()
                for move in moves:
                    X.append(s.featurize())
                    Y.append(util.one_hot(move))
                    A.append(score)
                    score = (-score) / DISCOUNT
                    s.move(*move)
        # TODO: advantage normalization
        X = np.array(X)
        Y = np.array(Y)
        A = np.array(A)
        a.pg_step(X, Y, A)
