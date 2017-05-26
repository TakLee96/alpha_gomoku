""" feature extractor for state """
import re
from collections import defaultdict


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
    features = defaultdict(lambda: 0)
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
        check_only(s, x, y, -dx, -dy, features)
    elif not bounded(negX, negY):
        check_only(s, x, y, dx, dy, features)
    elif state.board[posX, posY] == who and state.board[negX, negY] == who:
        check_connection(s, x, y, dx, dy, features)
    elif state.board[posX, posY] == (-who) and state.board[negX, negY] == (-who):
        check_disconnection(s, x, y, dx, dy, count)
    elif (state.board[posX, posY] == who and state.board[negX, negY] == (-who)) or
         (state.board[posX, posY] == (-who) and state.board[negX, negY] == who):
        check_only(s, x, y, dx, dy, features)
        check_only(s, x, y, -dx, -dy, features)
    elif state.board[posX, posY] == 0 and state.board[negX, negY] == 0:
        check_one_side(s, x, y, dx, dy, features)
    else:
        check_both_side(s, x, y, dx, dy, features)


def check_one_side(state, x, y, dx, dy, features):
    who = state.player
    llx, lly, rrx, rry = x - dx - dx, y - dy - dy, x + dx + dx, y + dy + dy
    ll = state.board[llx, lly] if bounded(llx, lly) else 0
    rr = state.board[rrx, rry] if bounded(rrx, rry) else 0
    one = "o" if who > 0 else "x"
    if ll == who:
        if rr == who:
            if ((not bounded(llx - dx, lly - dy)) or (state.board[llx - dx, lly - dy] != who)) and
               ((not bounded(rrx + dx, rry + dy)) or (state.board[rrx + dx, rry + dy] != who)):
                if who > 0:
                    features["o-o-o"] += 1
                else:
                    features["x-x-x"] += 1
            else:
                l = left_helper(s, llx + dx, lly + dy, dx, dy)
                r = right_helper(s, rrx - dx, rry - dy, dx, dy)
                new = format(l + "-" + one + "-" + r)
                if new is not None:
                    features[new] += 1
        else:
            l = left_helper(s, llx + dx, lly + dy, dx, dy)
            old = format(l + "-")
            new = format(l + "-" + one + "-")
            if old is not None:
                features[old] -= 1
            if new is not None:
                features[new] += 1
    elif rr == who:
        r = right_helper(s, rrx - dx, rry - dy, dx, dy)
        old = format("-" + r)
        new = format("-" + one + "-" + right)
        if old is not None:
            features[old] -= 1
        if new is not None:
            features[new] += 1
