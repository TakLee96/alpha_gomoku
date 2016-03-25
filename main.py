from bottle import route, run, request, static_file
from os import getcwd, path

from game import GameState
from brain import UNTSAgent

root = '/'.join(path.abspath(__file__).split('/')[:-1]) + '/public'
gamedata = dict()
agent = UNTSAgent()

@route('/', method='GET')
def callback():
    return static_file('index.html', root=root)

@route('/<path:path>', method='GET')
def callback(path):
    return static_file(path, root=root)

@route('/api', method='POST')
def callback():
    ip = request.environ.get('REMOTE_ADDR')
    if request.query.new:
        gamedata[ip] = GameState(int(request.query.first), [])
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
        nx, ny = agent.getAction(gamedata[ip])
        gamedata[ip].move((nx, ny))
        return {'x': nx, 'y': ny}
        
run(host='localhost', port='8000', debug=True)
