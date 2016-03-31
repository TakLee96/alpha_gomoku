import csv

class Counter(dict):
    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        return 0

def __check(state, who, move, direction, checked, features):
    x, y = move
    dx, dy = direction
    curr = state.moves[x][y]
    if curr == who:
        one = "o"
        other = "x"
    else:
        one = "x"
        other = "o"
    feature = one
    nx, ny = x + dx, y + dy
    jumped = False
    while state.inBound(nx, ny):
        next = state.moves[nx][ny]
        if next == curr:
            feature = feature + one
            checked.add(((nx, ny), direction))
            nx, ny = nx + dx, ny + dy
        elif next == state.EMPTY and state.inBound(nx+dx, ny+dy) and state.moves[nx+dx][ny+dy] == curr and not jumped:
            jumped = True
            feature = feature + "-"
            nx, ny = nx + dx, ny + dy
        else:
            if next == state.EMPTY:
                feature = feature + "-"
            else:
                feature = feature + other
            break
    if not state.inBound(nx, ny):
        feature = feature + other
    nx, ny = x - dx, y - dy
    jumped = False
    while state.inBound(nx, ny):
        next = state.moves[nx][ny]
        if next == curr:
            feature = one + feature
            checked.add(((nx, ny), direction))
            nx, ny = nx - dx, ny - dy
        elif next == state.EMPTY and state.inBound(nx-dx, ny-dy) and state.moves[nx-dx][ny-dy] == curr and not jumped:
            jumped = True
            feature = "-" + feature
            nx, ny = nx - dx, ny - dy
        else:
            if next == state.EMPTY:
                feature = "-" + feature
            else:
                feature = other + feature
            break
    if not state.inBound(nx, ny):
        feature = other + feature
    
    if 4 <= len(feature) <= 6 and (feature[0] == "-" or feature[-1] == "-"):
        if feature[0] == "-" and feature[-1] != "-":
            feature = feature[::-1]
        features[feature] = features[feature] + 1
    elif len(feature) == 7:
        feature = '*' + feature[1:6] + '*'
        features[feature] = features[feature] + 1
    elif len(feature) > 7 and '-xxxx-' in feature:
        features['long-death'] = features['long-death'] + 1
    elif len(feature) > 7 and '-oooo-' in feature:
        features['long-win'] = features['long-win'] + 1

def extractFeatures(state, who):
    checked = set()
    features = Counter()
    for move in state.hist:
        if move != None:
            for direction in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
                if (move, direction) not in checked:
                    checked.add((move, direction))
                    __check(state, who, move, direction, checked, features)
    return features

weight = Counter()
with open('weight.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        if len(row) > 1:
            weight[row[0].strip()] = float(row[1].strip())
INFINITY = weight['INFINITY']

def evaluate(features):
    value = 0.0
    for feature in features.keys():
        value += 1.0 * features[feature] * weight[feature]
    return value

def characterize(curr_features, features):
    total = 0
    for feature in set(curr_features.keys() + features.keys()):
        total += features[feature] - curr_features[feature]
    if total > 0:
        return "atk"
    if total < 0:
        return "def"
    return "ntr"