import pickle
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from sys import argv
from state import State
from feature import diff
from itertools import count
from os import path, listdir
from scipy.signal import convolve2d as conv2d


GAMES_PER_SAVE = 500
BUFFER_SIZE = 10000
BATCH_SIZE = 500
FILTER = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
]).astype(np.int8)


class TFRunner(mp.Process):
    def __init__(self, name, cmd_queue, action_queue, opponent=False):
        mp.Process.__init__(self)
        self.name = name
        self.cmd_queue = cmd_queue
        self.action_queue = action_queue
        self.opponent = opponent

    def run(self):
        with tf.Session() as session:
            root = path.join(path.dirname(__file__), "model", "reinforce", self.name)
            saver = tf.train.import_meta_graph(path.join(root, self.name + ".meta"), clear_devices=True)
            checkpoint = tf.train.latest_checkpoint(root)
            who = "agent"
            if self.opponent:
                who = "oppon"
                number = int(checkpoint.split("-")[-1]) / GAMES_PER_SAVE
                if number > 0:
                    number = np.random.randint(number + 1) * GAMES_PER_SAVE
                    checkpoint = "-".join(checkpoint.split("-")[:-1] + [str(number)])
            saver.restore(session, checkpoint)
            print("[TFRunner] %s %s loaded" % (who, checkpoint.split("/")[-1]))
            if not self.opponent:
                saver = tf.train.Saver(max_to_keep=99999999)
            cmd = self.cmd_queue.get()
            while cmd is not None:
                if "step" in cmd:
                    saver.save(session, path.join(root, self.name), global_step=cmd["step"], write_meta_graph=False)
                elif "x:0" in cmd and "y:0" in cmd and "f:0" in cmd:
                    session.run("train_step", feed_dict=cmd)
                elif "state" in cmd:
                    state = cmd["state"]
                    decision = np.random.random()
                    if self.opponent or decision < 0.96:
                        board = np.ndarray(shape=(1, 15, 15, 5), dtype=np.float32)
                        board[:, :, :, 0] = (state.board > 0)
                        board[:, :, :, 1] = (state.board < 0)
                        board[:, :, :, 2] = (state.board == 0)
                        board[:, :, :, 3] = 0
                        board[:, :, :, 4] = 1
                        prob = np.exp(session.run("prob:0", feed_dict={
                            "x:0": board,
                            "y:0": np.zeros(shape=(1, 225), dtype=np.float32),
                            "f:0": np.zeros(shape=1, dtype=np.float32)}).reshape(225))
                        best = prob.max()
                        good = set()
                        for i in range(225):
                            if prob[i] == best:
                                good.add(i)
                        if self.opponent:
                            x, y = np.unravel_index(np.random.choice(list(good)), dims=(15, 15))
                            result = (x, y, 0.1)
                        else:
                            chosen = np.random.choice(225, p=prob)
                            f = 0.1 if chosen in good else 1.0
                            x, y = np.unravel_index(chosen, dims=(15, 15))
                            result = (x, y, f)
                    else:
                        if len(state.history) == 0:
                            result = (7, 7, 0.1)
                        elif len(state.history) == 1:
                            x, y = np.unravel_index(np.random.choice([96, 97]), dims=(15, 15))
                            result = (x, y, 0.1)
                        else:
                            adjacent = conv2d(np.abs(state.board), FILTER, mode="same")
                            prob = np.logical_and(state.board == 0, adjacent > 0).astype(np.float32)
                            prob = (prob / prob.sum()).reshape(225)
                            x, y = np.unravel_index(np.random.choice(225, p=prob), dims=(15, 15))
                            result = (x, y, 4.0)
                    self.action_queue.put(result)
                else:
                    raise Exception("unknown command %r" % cmd)
                cmd = self.cmd_queue.get()
            print("[TFRunner] %s %s terminated" % (who, checkpoint.split("/")[-1]))


class Agent():
    def __init__(self, opponent=False):
        self.opponent = opponent
        self.black_cmd = mp.Queue()
        self.white_cmd = mp.Queue()
        self.action_queue = mp.Queue()
        self.black_runner = TFRunner("alphanet-5-black", self.black_cmd, self.action_queue, self.opponent)
        self.white_runner = TFRunner("alphanet-5-white", self.white_cmd, self.action_queue, self.opponent)
        self.black_runner.start()
        self.white_runner.start()

    def save(self, index):
        self.black_cmd.put({"step": index})
        self.white_cmd.put({"step": index})

    def get_action(self, state):
        if state.player == 1:
            self.black_cmd.put({"state": state})
        else:
            self.white_cmd.put({"state": state})
        return self.action_queue.get()

    def gradient_descent(self, tpl):
        black_X, black_y, black_f, white_X, white_y, white_f = tpl
        self.black_cmd.put({"x:0": black_X, "y:0": black_y, "f:0": black_f})
        self.white_cmd.put({"x:0": white_X, "y:0": white_y, "f:0": white_f})

    def refresh(self):
        assert self.opponent, "do not refresh agent"
        self.black_cmd.put(None)
        self.white_cmd.put(None)
        self.black_cmd = mp.Queue()
        self.white_cmd = mp.Queue()
        self.black_runner = TFRunner("alphanet-5-black", self.black_cmd, self.action_queue, self.opponent)
        self.white_runner = TFRunner("alphanet-5-white", self.white_cmd, self.action_queue, self.opponent)
        self.black_runner.start()
        self.white_runner.start()


def game_status(state):
    if state.end:
        if "violate" in state.features:
            return "black vio"
        elif state.player == 1:
            return "black win"
        else:
            return "white win"
    else:
        return "draw"


class ReplayBuffer:
    def __init__(self):
        self.black_board_buffer = np.ndarray(shape=(BUFFER_SIZE, 15, 15, 5), dtype=np.float32)
        self.white_board_buffer = np.ndarray(shape=(BUFFER_SIZE, 15, 15, 5), dtype=np.float32)
        self.black_action_buffer = np.ndarray(shape=BUFFER_SIZE, dtype=np.uint8)
        self.white_action_buffer = np.ndarray(shape=BUFFER_SIZE, dtype=np.uint8)
        self.black_feedback_buffer = np.ndarray(shape=BUFFER_SIZE, dtype=np.float32)
        self.white_feedback_buffer = np.ndarray(shape=BUFFER_SIZE, dtype=np.float32)
        self.black_size = 0
        self.white_size = 0
        self.black_index = 0
        self.white_index = 0

    def ready(self):
        return self.black_size >= BUFFER_SIZE and self.white_size >= BUFFER_SIZE

    def _black_append(self, board, action, feedback):
        b = np.ndarray(shape=(15, 15, 5), dtype=np.float32)
        b[:, :, 0] = (board > 0)
        b[:, :, 1] = (board < 0)
        b[:, :, 2] = (board == 0)
        b[:, :, 3] = 0
        b[:, :, 4] = 1
        self.black_board_buffer[self.black_index] = b
        self.black_action_buffer[self.black_index] = action
        self.black_feedback_buffer[self.black_index] = feedback
        self.black_size += 1
        self.black_index = (self.black_index + 1) % BUFFER_SIZE

    def _white_append(self, board, action, feedback):
        b = np.ndarray(shape=(15, 15, 5), dtype=np.float32)
        b[:, :, 0] = (board > 0)
        b[:, :, 1] = (board < 0)
        b[:, :, 2] = (board == 0)
        b[:, :, 3] = 0
        b[:, :, 4] = 1
        self.white_board_buffer[self.white_index] = b
        self.white_action_buffer[self.white_index] = action
        self.white_feedback_buffer[self.white_index] = feedback
        self.white_size += 1
        self.white_index = (self.white_index + 1) % BUFFER_SIZE

    def _add(self, moves, winner, feedbacks):
        if winner == 0: return
        assert len(moves) == len(feedbacks), "move feedback mismatch"
        board = np.zeros(shape=(15, 15), dtype=np.int8)
        multiplier = [0, 1.0, -0.5] if winner == 1 else [0, -0.5, 1.0]
        for i, (x, y) in enumerate(moves):
            if i % 2 == 0:
                self._black_append(np.copy(board),
                    np.ravel_multi_index((x, y), dims=(15, 15)),
                    multiplier[1]  * feedbacks[i])
                if board[x, y] != 0:
                    print("criticize %g" % (multiplier[1]  * feedbacks[i]))
                board[x, y] = 1
            else:
                self._white_append(np.copy(board),
                    np.ravel_multi_index((x, y), dims=(15, 15)),
                    multiplier[-1] * feedbacks[i])
                if board[x, y] != 0:
                    print("criticize %g" % (multiplier[-1]  * feedbacks[i]))
                board[x, y] = -1

    def add(self, state, feedbacks):
        moves = state.history
        winner = state.player if state.end else 0
        self._add(moves, winner, feedbacks)

    @staticmethod
    def _one_hot(y):
        n = len(y)
        y_h = np.zeros(shape=(n, 225), dtype=np.float32)
        for i in range(n):
            y_h[i, y[i]] = 1
        return y_h

    def sample(self):
        which_black = np.random.choice(BUFFER_SIZE, BATCH_SIZE, replace=False)
        which_white = np.random.choice(BUFFER_SIZE, BATCH_SIZE, replace=False)
        black_X = self.black_board_buffer[which_black, :, :, :]
        black_y = self._one_hot(self.black_action_buffer[which_black])
        black_f = self.black_feedback_buffer[which_black]
        white_X = self.white_board_buffer[which_white, :, :, :]
        white_y = self._one_hot(self.white_action_buffer[which_white])
        white_f = self.white_feedback_buffer[which_white]
        return black_X, black_y, black_f, white_X, white_y, white_f


def train(buff=ReplayBuffer(), start=0):
    agent = Agent()
    opponent = Agent(opponent=True)
    if start % GAMES_PER_SAVE != 0:
        opponent.refresh()
    bw_stat = [0, 0, 0]
    ao_stat = {"agent": 0, "oppon": 0, "the": 0}
    black_win, white_win, agent_win, oppon_win = [0] * 4
    for i in count(start):
        if i != 0 and i % GAMES_PER_SAVE == 0:
            print("[Stat] past %d games: [black %d white %d] [agent %d oppon %d]" % \
                (GAMES_PER_SAVE, bw_stat[1], bw_stat[-1], ao_stat["agent"], ao_stat["oppon"]))
            agent.save(i)
            print("[Model] checkpoint %d saved" % i)
            opponent.refresh()
            bw_stat = [0, 0, 0]
            ao_stat = {"agent": 0, "oppon": 0, "the": 0}
        state = State()
        if i % 2 == 0:
            players = [None, agent, opponent]
        else:
            players = [None, opponent, agent]
        feedbacks = list()
        while not state.end and len(state.history) != 225:
            x, y, f = players[state.player].get_action(state)
            try:
                state.move(x, y)
                if state.violate:
                    feedbacks.append(4.0)
                else:
                    feedbacks.append(f)
            except AssertionError:
                state.player = -state.player
                state.end = True
                state.history.append((x, y))
                feedbacks.append(4.0)
                print("[WARNING] game %d has a resign move with original feedback %g" % (i, f))
        if buff.ready():
            agent.gradient_descent(buff.sample())
        buff.add(state, feedbacks)
        winner = state.player if state.end else 0
        with open(path.join(path.dirname(__file__), "data", "reinforce", "%d.pkl" % i), "wb") as out:
            pickle.dump({ "history": state.history, "winner": winner, "feedbacks": feedbacks }, out)
        who = "the" if winner == 0 else ("oppon" if players[winner].opponent else "agent")            
        print("...... %d [%s %s] [black %d white %d]" % (i, who,
            game_status(state), buff.black_size, buff.white_size))
        bw_stat[winner] += 1
        ao_stat[who] += 1


def resume():
    buff = ReplayBuffer()
    root = path.join(path.dirname(__file__), "data", "reinforce")
    max_iter = 0
    print("[Resumer] loading historical game data")
    for f in listdir(root):
        if f.endswith(".pkl"):
            with open(path.join(root, f), "rb") as file:
                d = pickle.load(file)
                moves = d["history"]
                winner = d["winner"]
                feedbacks = d["feedbacks"]
                buff._add(moves, winner, feedbacks)
                max_iter = max(max_iter, int(f.split(".")[0]))
    print("[Resumer] loading complete, resume training")
    train(buff, max_iter+1)


if __name__ == "__main__":
    if len(argv) != 2 or argv[1] not in ("train", "resume"):
        print("Usage: python reinforce_policy.py [train/resume]")
    elif argv[1] == "train":
        train()
    elif argv[1] == "resume":
        resume()
    else:
        raise Exception("implementation error")
