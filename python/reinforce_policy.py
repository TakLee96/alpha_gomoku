import pickle
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from sys import argv
from state import State
from feature import diff
from itertools import count
from os import path, listdir


GAMES_PER_SAVE = 100
BUFFER_SIZE = 10000
BATCH_SIZE = 500


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
            if self.opponent:
                number = int(checkpoint.split("-")[-1]) / GAMES_PER_SAVE
                if number > 0:
                    number = np.random.randint(number + 1) * GAMES_PER_SAVE
                    checkpoint = "-".join(checkpoint.split("-")[:-1] + [str(number)])
            saver.restore(session, checkpoint)
            print("[TFRunner] %s loaded" % checkpoint.split("/")[-1])
            if not self.opponent:
                saver = tf.train.Saver(max_to_keep=99999999)
            cmd = self.cmd_queue.get()
            while cmd is not None:
                if "step" in cmd:
                    saver.save(session, path.join(root, self.name), global_step=cmd["step"], write_meta_graph=False)
                elif "x:0" in cmd and "y_:0" in cmd:
                    session.run("train_step", feed_dict=cmd)
                elif "state" in cmd:
                    state = cmd["state"]
                    if len(state.history) == 0:
                        self.action_queue.put((7, 7))
                    else:
                        prob = session.run("y:0", feed_dict={
                            "x:0": state.board.reshape(1, 225),
                            "y_:0": np.zeros(shape=(1, 225))
                        }).reshape(15, 15)
                        prob = np.exp(prob)
                        prob = prob / prob.sum()
                        four_x_count = state.features["four-x"] + state.features["-xxxxo"]
                        three_x_count = state.features["-x-xx-"] + state.features["-xxx-"]
                        four_o_count = state.features["four-o"] + state.features["-oooox"]
                        three_o_count = state.features["-o-oo-"] + state.features["-ooo-"]
                        for j in range(15):
                            for k in range(15):
                                new, old = diff(state, j, k)
                                if state.board[j, k] != 0 or new["-o-oo-"] + new["-ooo-"] >= 2 or \
                                    new["four-o"] + new["-oooo-"] + new["-oooox"] >= 2 or state._long(j, k):
                                    prob[j, k] = 0
                                elif new["win-o"] > 0 or new["win-x"] > 0:
                                    prob[j, k] = 10000 * prob[j, k]
                                else:
                                    if state.player == 1:
                                        if four_x_count > 0 and four_x_count - old["four-x"] - old["-xxxxo"] <= 0:
                                            prob[j, k] = 1000 * prob[j, k]
                                        elif new["-oooo-"] > 0:
                                            prob[j, k] = 100 * prob[j, k]
                                        elif three_x_count > 0 and (three_x_count - old["-x-xx-"] - old["-xxx-"] <= 0 or
                                            new["four-o"] + new["-oooox"] > 0):
                                            prob[j, k] = 10 * prob[j, k]
                                    else:
                                        if four_o_count > 0 and four_o_count - old["four-o"] - old["-oooox"] <= 0:
                                            prob[j, k] = 1000 * prob[j, k]
                                        elif new["-xxxx-"] > 0:
                                            prob[j, k] = 100 * prob[j, k]
                                        elif three_o_count > 0 and (three_o_count - old["-o-oo-"] - old["-ooo-"] <= 0 or
                                            new["four-x"] + new["-xxxxo"] > 0):
                                            prob[j, k] = 10 * prob[j, k]
                        prob = (prob / prob.sum()).reshape(225)
                        action = np.unravel_index(np.random.choice(225, p=prob), (15, 15))
                        self.action_queue.put(action)
                else:
                    raise Exception("unknown command %r" % cmd)
                cmd = self.cmd_queue.get()
            print("[TFRunner] %s terminated" % checkpoint.split("/")[-1])


class Agent():
    def __init__(self, opponent=False):
        self.opponent = opponent
        self.black_cmd = mp.Queue()
        self.white_cmd = mp.Queue()
        self.action_queue = mp.Queue()
        self.black_runner = TFRunner("gonet-black", self.black_cmd, self.action_queue, self.opponent)
        self.white_runner = TFRunner("gonet-white", self.white_cmd, self.action_queue, self.opponent)
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
        black_X, black_y, white_X, white_y = tpl
        self.black_cmd.put({"x:0": black_X, "y_:0": black_y})
        self.white_cmd.put({"x:0": white_X, "y_:0": white_y})

    def refresh(self):
        self.black_cmd.put(None)
        self.white_cmd.put(None)
        self.black_cmd = mp.Queue()
        self.white_cmd = mp.Queue()
        self.black_runner = TFRunner("gonet-black", self.black_cmd, self.action_queue, self.opponent)
        self.white_runner = TFRunner("gonet-white", self.white_cmd, self.action_queue, self.opponent)
        self.black_runner.start()
        self.white_runner.start()        


def game_status(state):
    if state.end:
        if state.player == 1:
            return "black win"
        else:
            return "white win"
    else:
        return "draw"


class ReplayBuffer:
    def __init__(self):
        self.black_board_buffer = np.ndarray(shape=(BUFFER_SIZE, 225), dtype=np.int8)
        self.white_board_buffer = np.ndarray(shape=(BUFFER_SIZE, 225), dtype=np.int8)
        self.black_action_buffer = np.ndarray(shape=BUFFER_SIZE, dtype=np.uint8)
        self.white_action_buffer = np.ndarray(shape=BUFFER_SIZE, dtype=np.uint8)
        self.black_size = 0
        self.white_size = 0
        self.black_index = 0
        self.white_index = 0

    def ready(self):
        return self.black_size >= BUFFER_SIZE and self.white_size >= BUFFER_SIZE

    def _black_append(self, board, action):
        self.black_board_buffer[self.black_index] = board.reshape(225)
        self.black_action_buffer[self.black_index] = action
        self.black_size += 1
        self.black_index = (self.black_index + 1) % BUFFER_SIZE

    def _white_append(self, board, action):
        self.white_board_buffer[self.white_index] = board.reshape(225)
        self.white_action_buffer[self.white_index] = action
        self.white_size += 1
        self.white_index = (self.white_index + 1) % BUFFER_SIZE

    def _add(self, moves, winner):
        state = State()
        if winner == 1:
            for i, (x, y) in enumerate(moves):
                if i % 2 == 0:
                    self._black_append(np.copy(state.board),
                        np.ravel_multi_index((x, y), dims=(15, 15)))
                state.move(x, y)
        elif winner == -1:
            for i, (x, y) in enumerate(moves):
                if i % 2 == 1:
                    self._white_append(np.copy(state.board),
                        np.ravel_multi_index((x, y), dims=(15, 15)))
                state.move(x, y)

    def add(self, state):
        moves = state.history
        winner = state.player if state.end else 0
        self._add(moves, winner)

    @staticmethod
    def _one_hot(y):
        n = len(y)
        y_h = np.zeros(shape=(n, 225), dtype=float)
        for i in range(n):
            y_h[i, y[i]] = 1
        return y_h

    def sample(self):
        which_black = np.random.choice(BUFFER_SIZE, BATCH_SIZE, replace=False)
        which_white = np.random.choice(BUFFER_SIZE, BATCH_SIZE, replace=False)
        black_X = self.black_board_buffer[which_black, :]
        black_y = self._one_hot(self.black_action_buffer[which_black])
        white_X = self.white_board_buffer[which_white, :]
        white_y = self._one_hot(self.white_action_buffer[which_white])
        return black_X, black_y, white_X, white_y


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
        while not state.end and len(state.history) != 225:
            x, y = players[state.player].get_action(state)
            state.move(x, y)
        if buff.ready():
            agent.gradient_descent(buff.sample())
        buff.add(state)
        winner = state.player if state.end else 0
        with open(path.join(path.dirname(__file__), "data", "reinforce", "%d.pkl" % i), "wb") as out:
            pickle.dump({"history": state.history, "winner": winner }, out)
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
                buff._add(moves, winner)
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
