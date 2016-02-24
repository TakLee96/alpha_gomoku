from bottle import route, run, request, static_file
from os import getcwd, path, environ
from brain import GameData, AGENTS

root = '/'.join(path.abspath(__file__).split('/')[:-1]) + '/public'

@route('/', method='GET')
def callback():
    return static_file('index.html', root=root)

@route('/<path:path>', method='GET')
def callback(path):
    return static_file(path, root=root)

gamedata = dict()

@route('/api', method='POST')
def callback():
    ip = request.environ.get('REMOTE_ADDR')
    if request.query.new:
        gamedata[ip] = GameData(int(request.query.first), agent=AGENTS[request.query.agent]()) 
        if gamedata[ip].first == GameData.AI:
            nx, ny = GameData.GRID_SIZE/2, GameData.GRID_SIZE/2
            gamedata[ip] = gamedata[ip].generateSuccessor(nx, ny)
            return {'x': nx, 'y': ny}
    elif request.query.end:
        if ip in gamedata:
            del gamedata[ip]
    else:
        x = int(request.query.x)
        y = int(request.query.y)
        gamedata[ip] = gamedata[ip].generateSuccessor(x, y)
        (nx, ny) = gamedata[ip].think()
        gamedata[ip] = gamedata[ip].generateSuccessor(nx, ny)
        return {'x': nx, 'y': ny}

run(host=environ['host'], port=environ['port'], debug=True)
