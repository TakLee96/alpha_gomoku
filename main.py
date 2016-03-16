from bottle import route, run, request, static_file
from os import getcwd, path, environ
from game import GameState
from agent import AlphaBetaAgent, ReflexAgent
from learn import ApproximateQLearningAgent

AGENTS = {'m': AlphaBetaAgent(), 'r': ReflexAgent(), 'l': ApproximateQLearningAgent()}

root = '/'.join(path.abspath(__file__).split('/')[:-1]) + '/public'

@route('/', method='GET')
def callback():
    return static_file('index.html', root=root)

@route('/<path:path>', method='GET')
def callback(path):
    return static_file(path, root=root)

gamedata = dict()
agent = dict()

@route('/api', method='POST')
def callback():
    ip = request.environ.get('REMOTE_ADDR')
    if request.query.new:
        gamedata[ip] = GameState(int(request.query.first))
        agent[ip] = request.query.agent
        if gamedata[ip].first == GameState.AI:
            nx, ny = GameState.GRID_SIZE/2, GameState.GRID_SIZE/2
            gamedata[ip].move((nx, ny))
            return {'x': nx, 'y': ny}
    elif request.query.end:
        if ip in gamedata:
            del gamedata[ip]
    else:
        x = int(request.query.x)
        y = int(request.query.y)
        gamedata[ip].move((x, y))
        action = AGENTS[agent[ip]].getAction(gamedata[ip])
        gamedata[ip].move(action)
        return {'x': action[0], 'y': action[1]}
        
        

run(host=environ['host'], port=environ['port'], debug=True)
