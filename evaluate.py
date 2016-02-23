def normalize(x, y, ax, ay, bx, by, who, state):
    seq = [0 for _ in range(9)]
    seq[4] = who
    for i in range(1, 5):
        nx, ny = x + i * ax, y + i * ay
        if state.inBound(nx, ny):
            seq[4-i] = state.moves[nx][ny]
        else:
            seq[4-i] = other(who)
    for i in range(1, 5):
        nx, ny = x + i * bx, y + i * by
        if state.inBound(nx, ny):
            seq[4+i] = state.moves[nx][ny]
        else:
            seq[4+i] = other(who)
    return seq

def length(seq):
    who = seq[4]
    copy = seq[:]
    left = 4
    right = 4
    for i in range(5, 9):
        if seq[i] != who:
            break
        right = i
    for i in range(3, -1, -1):
        if seq[i] != who:
            break
        left = i
    return (left, right, right - left + 1)

def countEmpty(seq):
    who = seq[4]
    leftCount = 0
    rightCount = 0
    for i in range(5, 9):
        if seq[i] == EMPTY:
            leftCount += 1
        elif seq[i] == other(who):
            break
    for i in range(3, -1, -1):
        if seq[i] == EMPTY:
            rightCount += 1
        elif seq[i] == other(who):
            break
    return (leftCount, rightCount)

def checkfour(seq, left, right):
    leftCount, rightCount = self.countEmpty(seq)
    if leftCount != 0 and rightCount != 0:
        return DEATH
    elif leftCount != 0 or rightCount != 0:
        return URGENT
    return USELESS

def checkthree(seq, left, right):
    leftCount, rightCount = self.countEmpty(seq)
    who = seq[4]
    if (seq[left-1] == EMPTY and seq[left-2] == who):
        return URGENT
    if (seq[right+1] == EMPTY and seq[right+2] == who):
        return URGENT
    if leftCount != 0 and rightCount != 0:
        if leftCount + rightCount > 2:
            return URGENT
        else:
            return IMPORTANT
    elif leftCount != 0 or rightCount != 0:
        if leftCount + rightCount > 2:
            return IMPORTANT
    return USELESS

def checktwo(seq, left, right):
    leftCount, rightCount = self.countEmpty(seq)
    who = seq[4]
    # TODO: will 2 jump 2 or 1 jump 2 jump 1 ever occur
    if (seq[left-1] == EMPTY and seq[left-2] == who):
        return self.checkthree([ other(who) ] + seq[:left-1] + seq[left:], left-1, right)
    if (seq[right+1] == EMPTY and seq[right+2] == who):
        return self.checkthree(seq[:right+1] + seq[right+2:] + [ other(who) ], left, right+1)
    if leftCount == 0 and rightCount == 0:
        return USELESS
    if leftCount == 0 or rightCount == 0:
        return OKAY
    if leftCount + rightCount < 3:
        return USELESS
    return GOOD

def score(seq):
    who = seq[4]
    left, right, length = length(seq)
    if length >= 5:
        return DEATH
    elif length == 4:
        return checkfour(seq, left, right)
    elif length == 3:
        return checkthree(seq, left, right)
    elif length == 2:
        return checktwo(seq, left, right)
    elif length == 1:
        return USELESS # TODO: can improve by looking at jumping 2

def evaluate(state):
    score = 0
    for x, y, who in state.hist:
        temp  = score(normalize(x, y, 1, 0, -1, 0, who, state))
        temp += score(normalize(x, y, 0, 1, 0, -1, who, state))
        temp += score(normalize(x, y, 1, 1, -1, -1, who, state))
        temp += score(normalize(x, y, 1, -1, -1, 1, who, state))
        multiplier = 1.0 if who == AI else -20.1
        score += multiplier * temp
    return score + random()