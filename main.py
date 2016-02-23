from bottle import route, run, request, static_file
from os import getcwd, path
from brain import GameData

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
        gamedata[ip] = GameData(int(request.query.first))
    elif request.query.end:
        del gamedata[ip]
    else:
        x = int(request.query.x)
        y = int(request.query.y)
        gamedata[ip] = gamedata[ip].update(x, y, 2)
        (nx, ny) = gamedata[ip].think()
        gamedata[ip] = gamedata[ip].update(nx, ny, 1)
        return {'x': nx, 'y': ny}
 
run(host='localhost', port='8000', debug=True)
