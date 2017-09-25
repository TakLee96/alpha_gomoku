import pickle
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from sys import argv
from state import State
from itertools import count
from os import path, listdir


GAMES_PER_SAVE = 50
BUFFER_SIZE = 10000
BATCH_SIZE = 50


class TFRunner(mp.Process):
    def __init__(self, name, state_queue, action_queue):
        mp.Process.__init__(self)
        self.name = name
        self.state_queue = state_queue
        self.action_queue = action_queue

    def run(self):
        with tf.Session() as session:
            root = path.join(path.dirname(__file__), "model", "reinforce", self.name)
            saver = tf.train.import_meta_graph(path.join(root, self.name + ".meta"), clear_devices=True)
            saver.restore(session, tf.train.latest_checkpoint(root))
            while True:
                state = self.state_queue.get()
                if state is None:
                    break
                elif isinstance(state, dict):
                    session.run("train_step", feed_dict=state)
                elif isinstance(state, str):
                    saver.save(session, path.join(root, self.name), global_step=int(state), write_meta_graph=False)
                else:
                    score = np.ndarray(shape=(15, 15), dtype=float)
                    for x in range(15):
                        for y in range(15):
                            if state.board[x, y] == 0:
                                state.move(x, y)
                                if state.end:
                                    score[x, y] = state.player
                                else:
                                    score[x, y] = session.run("y:0", feed_dict={
                                        "x:0": state.board.reshape(1, 225),
                                        "y_:0": np.zeros(shape=(1, 1))})
                                state.rewind()
                            else:
                                score[x, y] = -state.player
                    score = score.reshape(225) * state.player
                    if np.random.random() > 0.8:
                        score = np.square(score + 1)
                        score = score / score.sum()
                        self.action_queue.put(np.unravel_index(np.random.choice(225, p=score), (15, 15)))
                    else:
                        self.action_queue.put(np.unravel_index(score.argmax(), (15, 15)))


class Agent():
    def __init__(self):
        self.black_state = mp.Queue()
        self.white_state = mp.Queue()
        self.action_queue = mp.Queue()
        self.black_runner = TFRunner("qbtnet-black", self.black_state, self.action_queue)
        self.white_runner = TFRunner("qbtnet-white", self.white_state, self.action_queue)
        self.black_runner.start()
        self.white_runner.start()

    def save(self, index):
        self.black_state.put(str(index))
        self.white_state.put(str(index))

    def get_action(self, state):
        if state.player == -1:
            self.black_state.put(state)
        else:
            self.white_state.put(state)
        return self.action_queue.get()

    def gradient_descent(self, tpl):
        black_X, black_y, white_X, white_y = tpl
        self.black_state.put({"x:0": black_X, "y_:0": black_y})
        self.white_state.put({"x:0": white_X, "y_:0": white_y})


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
        self.black_board_buffer = np.ndarray(shape=(BUFFER_SIZE, 225), dtype=float)
        self.white_board_buffer = np.ndarray(shape=(BUFFER_SIZE, 225), dtype=float)
        self.black_score_buffer = np.ndarray(shape=(BUFFER_SIZE, 1), dtype=float)
        self.white_score_buffer = np.ndarray(shape=(BUFFER_SIZE, 1), dtype=float)
        self.black_size = 0
        self.white_size = 0
        self.black_index = 0
        self.white_index = 0

    def ready(self):
        return self.black_size >= BUFFER_SIZE and self.white_size >= BUFFER_SIZE

    def _black_append(self, board, score):
        self.black_board_buffer[self.black_index] = board.reshape(225)
        self.black_score_buffer[self.black_index] = score
        self.black_size += 1
        self.black_index = (self.black_index + 1) % BUFFER_SIZE

    def _white_append(self, board, score):
        self.white_board_buffer[self.white_index] = board.reshape(225)
        self.white_score_buffer[self.white_index] = score
        self.white_size += 1
        self.white_index = (self.white_index + 1) % BUFFER_SIZE

    def _add(self, moves, winner):
        state = State()
        self._black_append(np.copy(state.board), winner)
        for i, (x, y) in enumerate(moves):
            state.move(x, y)
            board = np.copy(state.board)
            if i % 2 == 1:
                self._black_append(board, winner)
            else:
                self._white_append(board, winner)

    def add(self, state):
        moves = state.history
        winner = state.player if state.end else 0
        self._add(moves, winner)

    def sample(self):
        which_black = np.random.choice(BUFFER_SIZE, BATCH_SIZE, replace=False)
        which_white = np.random.choice(BUFFER_SIZE, BATCH_SIZE, replace=False)
        black_X = self.black_board_buffer[which_black, :]
        black_y = self.black_score_buffer[which_black, :]
        white_X = self.white_board_buffer[which_white, :]
        white_y = self.white_score_buffer[which_white, :]
        return black_X, black_y, white_X, white_y


def train(buff=ReplayBuffer(), start=0):
    agent = Agent()
    for i in count(start):
        if buff.ready() and i % GAMES_PER_SAVE == 0:
            agent.save(i)
            print("model checkpoint %d" % i)
        state = State()
        num_updates = 0
        while not state.end and len(state.history) != 225:
            x, y = agent.get_action(state)
            state.move(x, y)
            if buff.ready():
                agent.gradient_descent(buff.sample())
                num_updates += 1
        buff.add(state)
        with open(path.join(path.dirname(__file__), "data", "reinforce", "%d.pkl" % i), "wb") as out:
            pickle.dump({
                "history": state.history,
                "winner": state.player if state.end else 0 }, out)
        print("%d %s [black %d white %d] [%d updates] [game saved]" % (i, game_status(state),
            buff.black_size, buff.white_size, num_updates))


def resume():
    buff = ReplayBuffer()
    root = path.join(path.dirname(__file__), "data", "reinforce")
    max_iter = 0
    print("loading historical game data")
    for f in listdir(root):
        with open(path.join(root, f), "rb") as file:
            d = pickle.load(file)
            moves = d["history"]
            winner = d["winner"]
            buff._add(moves, winner)
            max_iter = max(max_iter, int(f.split(".")[0]))
    print("loading complete, resume training")
    train(buff, max_iter+1)


def help():
    print("Usage: python reinforce_value.py [train/resume]")


if __name__ == "__main__":
    if len(argv) != 2 or argv[1] not in ("train", "resume"):
        print("Usage: python reinforce_value.py [train/resume]")
    elif argv[1] == "train":
        train()
    elif argv[1] == "resume":
        resume()
    else:
        raise Exception("implementation error")
