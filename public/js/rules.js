var rules = (function () { var exports = {};

var resolve = {
    top:         function (i, j) { return [ i - 1, j     ]; },
    bottom:      function (i, j) { return [ i + 1, j     ]; },
    left:        function (i, j) { return [ i    , j - 1 ]; },
    right:       function (i, j) { return [ i    , j + 1 ]; },
    topleft:     function (i, j) { return [ i - 1, j - 1 ]; },
    topright:    function (i, j) { return [ i - 1, j + 1 ]; },
    bottomleft:  function (i, j) { return [ i + 1, j - 1 ]; },
    bottomright: function (i, j) { return [ i + 1, j + 1 ]; }
};

var outOfBound = function (i, j) {
    return i < 0 || i >= GRID_SIZE || j < 0 || j >= GRID_SIZE;
};

var countDirection = function (i, j, state, direction, player) {
    if (outOfBound(i, j) || state[i][j].player != player)
        return 0;
    var next = resolve[direction](i, j);
    return 1 + countDirection(next[0], next[1], state, direction, player);
};

exports.win = function (i, j, state) {
    var player = state[i][j].player;
    if (
        countDirection(i, j, state, 'top', player) + countDirection(i, j, state, 'bottom', player) - 1 == 5 ||
        countDirection(i, j, state, 'left', player) + countDirection(i, j, state, 'right', player) - 1 == 5 ||
        countDirection(i, j, state, 'topleft', player) + countDirection(i, j, state, 'bottomright', player) - 1 == 5 ||
        countDirection(i, j, state, 'topright', player) + countDirection(i, j, state, 'bottomleft', player) - 1 == 5
    ) return true;
    return false;
};


return exports; })();
