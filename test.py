from game import GameState
from brain import AlphaBetaAgent, default_heuristic
from os import environ

participants = [
    { 'name': "3-None",    'depth': 3, 'heuristic': None                            },
    { 'name': "6-Normal",  'depth': 6, 'heuristic': default_heuristic               },
    { 'name': "8-Sparse",  'depth': 8, 'heuristic': lambda d: 4 if d < 5 else 9 - d },
    { 'name': "8-Average", 'depth': 8, 'heuristic': lambda d: 5                     },
]

def pairwise(arr):
    pairs = []
    for i in range(len(arr)-1):
        for j in range(i+1, len(arr)):
            pairs.append((arr[i], arr[j]))
    return pairs

def construct(config, index):
    return AlphaBetaAgent(index=index, depth=config['depth'], heuristic=config['heuristic'])

pairs = pairwise(participants)

results = dict()

for (ca, cb) in pairs:
    a = construct(ca, GameState.AI)
    b = construct(cb, GameState.HUMAN)
    a_first_win = 0
    a_last_win  = 0
    b_first_win = 0
    b_last_win  = 0
    a_b_tie     = 0
    for i in range(8):
        state = GameState(i % 2 + 1, [])
        print state.hist,
        players = [a, b]
        j = i % 2
        while not state.isTerminal():
            curr = players[j]
            move = curr.getAction(state)
            state.move(move)
            j = (j + 1) % 2
        if state.isWin(a.index):
            if a.index == state.first:
                a_first_win += 1
            else:
                a_last_win += 1
        elif state.isWin(b.index):
            if b.index == state.first:
                b_first_win += 1
            else:
                b_last_win += 1
        else:
            a_b_tie += 1
        print state.hist
    results[ca['name'] + "(a) vs " + cb['name'] + "(b)"] = {
        'afw': a_first_win,
        'alw': a_last_win,
        'bfw': b_first_win,
        'blw': b_last_win,
        'tie': a_b_tie,
    }

print "===================================================="
print results