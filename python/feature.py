""" feature extractor for state """
import re


class defaultdict(dict):
    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)
        return 0

    def __setitem__(self, key, val):
        if key is not None:
            dict.__setitem__(self, key, val)

    def add(self, other):
        for key, val in other.items():
            self[key] += val

    def sub(self, other):
        for key, val in other.items():
            self[key] -= val


directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
fundamental = {
    "-xxo", "-x-xo", "-oox", "-o-ox",
    "-x-x-", "-xx-", "-o-o-", "-oo-",
    "-x-xxo", "-xxxo", "-o-oox", "-ooox",
    "-x-xx-", "-xxx-", "-o-oo-", "-ooo-",
    "-xxxx-", "-xxxxo", "-oooo-", "-oooox",
}
black_four = re.compile(".*[x-](o-ooo|oo-oo|ooo-o)[x-].*");
white_four = re.compile(".*[o-](x-xxx|xx-xx|xxx-x)[o-].*");
jump_three = re.compile(".[xo][xo]-[xo].");


def format(feature):
    if len(feature) < 3:
        return None
    if "oooooo" in feature or "xxxxxx" in feature:
        return None
    if "ooooo" in feature:
        return "win-o"
    if "xxxxx" in feature:
        return "win-x"
    if black_four.match(feature):
        return "four-o"
    if white_four.match(feature):
        return "four-x"
    if len(feature) > 6:
        return None
    if feature[0] != "-" and feature[-1] == "-":
        feature = feature[::-1]
    if jump_three.match(feature):
        feature = feature[0] + feature[-2:0:-1] + feature[-1]
    if feature in fundamental:
        return feature
    return None


def diff(state, x, y):
    features = defaultdict()
    for dx, dy in directions:
        check(state, x, y, dx, dy, features)
    return features


def bounded(x, y):
    return 0 <= x <= 14 and 0 <= y <= 14


def check(state, x, y, dx, dy, features):
    posX, posY, negX, negY = x + dx, y + dy, x - dx, y - dy
    who = state.player
    if (not bounded(posX, posY)) and (not bounded(negX, negY)):
        return
    if not bounded(posX, posY):
        check_only(state, x, y, -dx, -dy, features)
    elif not bounded(negX, negY):
        check_only(state, x, y, dx, dy, features)
    elif state.board[posX, posY] == who and state.board[negX, negY] == who:
        check_connection(state, x, y, dx, dy, features)
    elif state.board[posX, posY] == (-who) and state.board[negX, negY] == (-who):
        check_disconnection(state, x, y, dx, dy, features)
    elif (state.board[posX, posY] == who and state.board[negX, negY] == (-who)) or \
         (state.board[posX, posY] == (-who) and state.board[negX, negY] == who):
        check_only(state, x, y, dx, dy, features)
        check_only(state, x, y, -dx, -dy, features)
    elif state.board[posX, posY] == 0 and state.board[negX, negY] == 0:
        check_one_side(state, x, y, dx, dy, features)
    else:
        check_both_side(state, x, y, dx, dy, features)


def check_one_side(state, x, y, dx, dy, features):
    who = state.player
    llx, lly, rrx, rry = x - dx - dx, y - dy - dy, x + dx + dx, y + dy + dy
    ll = state.board[llx, lly] if bounded(llx, lly) else 0
    rr = state.board[rrx, rry] if bounded(rrx, rry) else 0
    one = "o" if who > 0 else "x"
    if ll == who:
        if rr == who:
            if ((not bounded(llx - dx, lly - dy)) or (state.board[llx - dx, lly - dy] != who)) and \
               ((not bounded(rrx + dx, rry + dy)) or (state.board[rrx + dx, rry + dy] != who)):
                if who > 0:
                    features["o-o-o"] += 1
                else:
                    features["x-x-x"] += 1
            else:
                left = left_helper(state, llx + dx, lly + dy, dx, dy)
                right = right_helper(state, rrx - dx, rry - dy, dx, dy)
                new = format(left + "-" + one + "-" + right)
                features[new] += 1
        else:
            left = left_helper(state, llx + dx, lly + dy, dx, dy)
            old = format(left + "-")
            new = format(left + "-" + one + "-")
            features[old] -= 1
            features[new] += 1
    elif rr == who:
        right = right_helper(state, rrx - dx, rry - dy, dx, dy)
        old = format("-" + right)
        new = format("-" + one + "-" + right)
        features[old] -= 1
        features[new] += 1


def check_both_side(state, x, y, dx, dy, features):
    if state.board[x + dx, y + dy] == 0:
        return check_both_side(state, x, y, -dx, -dy, features)
    who = state.player
    one = "o" if who > 0 else "x"
    other = "x" if who > 0 else "o"
    llx, lly = x - dx - dx, y - dy - dy
    ll = state.board[llx, lly] if bounded(llx, lly) else 0
    if state.board[x + dx, y + dy] == who:
        if ll == who:
            left = left_helper(state, llx + dx, lly + dy, dx, dy)
            right = right_helper(state, x, y, dx, dy)
            new = format(left + "-" + one + right)
            features[new] += 1
        else:
            right = right_helper(state, x, y, dx, dy)
            old = format("-" + right)
            new = format("-" + one + right)
            features[old] -= 1
            features[new] += 1
    else:
        if ll == who:
            left = left_helper(state, llx + dx, lly + dy, dx, dy)
            right = right_helper(state, llx + dx, lly + dy, dx, dy)
            old = format("-" + right)
            new1 = format(one + right)
            new2 = format(left + "-" + one + other)
            features[old] -= 1
            features[new1] += 1
            features[new2] += 1
        else:
            right = right_helper(state, x, y, dx, dy)
            old = format("-" + right)
            new = format(one + right)
            features[old] -= 1
            features[new] += 1


def left_helper(state, x, y, dx, dy):
    nx, ny = x - dx, y - dy
    who = state.board[nx, ny]
    one = "o" if who > 0 else "x"
    other = "x" if who > 0 else "o"
    feature = ""
    while bounded(nx, ny):
        next = state.board[nx, ny]
        if next == who:
            feature = one + feature
            nx, ny = nx - dx, ny - dy
        elif next == 0 and bounded(nx - dx, ny - dy) and state.board[nx - dx, ny - dy] == who:
            feature = "-" + feature
            nx, ny = nx - dx, ny - dy
        else:
            if next == 0:
                feature = "-" + feature
            else:
                feature = other + feature
            break
    if not bounded(nx, ny):
        feature = other + feature
    return feature


def right_helper(state, x, y, dx, dy):
    nx, ny = x + dx, y + dy
    who = state.board[nx, ny]
    one = "o" if who > 0 else "x"
    other = "x" if who > 0 else "o"
    feature = ""
    while bounded(nx, ny):
        next = state.board[nx, ny]
        if next == who:
            feature = feature + one
            nx, ny = nx + dx, ny + dy
        elif next == 0 and bounded(nx + dx, ny + dy) and state.board[nx + dx, ny + dy] == who:
            feature = feature + "-"
            nx, ny = nx + dx, ny + dy
        else:
            if next == 0:
                feature = feature + "-"
            else:
                feature = feature + other
            break
    if not bounded(nx, ny):
        feature = feature + other
    return feature


def check_connection(state, x, y, dx, dy, features):
    left = left_helper(state, x, y, dx, dy)
    right = right_helper(state, x, y, dx, dy)
    one = "o" if state.player > 0 else "x"
    old = format(left + "-" + right)
    new = format(left + one + right)
    features[old] -= 1
    features[new] += 1


def check_disconnection(state, x, y, dx, dy, features):
    left = left_helper(state, x, y, dx, dy)
    right = right_helper(state, x, y, dx, dy)
    one = "o" if state.player > 0 else "x"
    old = format(left + "-" + right)
    features[old] -= 1
    new_left = format(left + one)
    new_right = format(one + right)
    features[new_left] += 1
    features[new_right] += 1


def check_only(state, x, y, dx, dy, features):
    nx, ny = x + dx, y + dy
    who = state.player
    if bounded(nx, ny) and state.board[nx, ny] != 0:
        right = right_helper(state, x, y, dx, dy)
    elif bounded(nx + dx, ny + dy) and state.board[nx + dx, ny + dy] == who:
        right = "-" + right_helper(state, nx, ny, dx, dy)
        nx, ny = nx + dx, ny + dy
    else:
        return
    if who == 1:
        if state.board[nx, ny] == 1:
            old = "-" + right
            new = "xo" + right
        else:
            old = "-" + right
            new = "o" + right
    else:
        if state.board[nx, ny] == 1:
            old = "-" + right
            new = "x" + right
        else:
            old = "-" + right
            new = "ox" + right
    old = format(old)
    new = format(new)
    features[old] -= 1
    features[new] += 1
