from game import GameState
from brain import AlphaBetaAgent

a = AlphaBetaAgent(GameState.AI, 3)
b = AlphaBetaAgent(GameState.HUMAN, 3)

def other(agent):
    if agent == a:
        return b
    return a

aw = 0
bw = 0

for i in range(10):
    print "begin"
    print (GameState.GRID_SIZE/2, GameState.GRID_SIZE/2)
    state = GameState(GameState.AI)
    state.move((GameState.GRID_SIZE/2, GameState.GRID_SIZE/2))
    agent = b
    while not state.isTerminal():
        action = agent.getAction(state)
        print action
        state.move(action)
        agent = other(agent)

    print "end"
    if state.isWin(GameState.AI):
        print "a wins"
        aw += 1
    else:
        print "b wins"
        bw += 1

print "a wins", aw, "vs b wins", bw

print state.hist