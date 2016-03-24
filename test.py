from game import GameState
from brain import AlphaBetaAgent
from os import environ

times = 10
if 'times' in environ:
    times = int(environ['times'])


a = AlphaBetaAgent(GameState.AI, 4, False)
b = AlphaBetaAgent(GameState.HUMAN, 6, True)

def other(agent):
    if agent == a:
        return b
    return a

aw = 0
bw = 0

for i in range(times):
    print "begin"
    hist = []
    hist.append((GameState.GRID_SIZE/2, GameState.GRID_SIZE/2))
    state = GameState(GameState.AI)
    state.move((GameState.GRID_SIZE/2, GameState.GRID_SIZE/2))
    agent = b
    print hist[0]
    while not state.isTerminal():
        action = agent.getAction(state)
        print action
        hist.append(action)
        state.move(action)
        agent = other(agent)
    print "end"
    print hist
    if state.isWin(GameState.AI):
        print "a wins"
        aw += 1
    else:
        print "b wins"
        bw += 1

print "a wins", aw, "vs b wins", bw

print state.hist