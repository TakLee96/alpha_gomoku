(function () {
    var player = 1; // 0 for empty, 1 for blue and 2 for green
    var other = function (player) { return 3 - player };
    var name = function (player) { return (player == 1) ? 'blue' : 'green';  }

    var Grid = function () {
        this.player = 0;
    };
    Grid.prototype.image = function () {
        switch (this.player) {
            case 1: return "./img/grid-blue.jpg";
            case 2: return "./img/grid-green.jpg";
            default: return "./img/grid-empty.jpg";
        };
    };
    Grid.prototype.play = function (player) {
        this.player = player;
    }
    Grid.prototype.isEmpty = function () {
        return this.player == 0;
    };

    var state = [], history = [], row = null, grid = null;
    var board = "<table>";
    for (var i = 0; i < GRID_SIZE; i++) {
        board += "<tr>"
        state[i] = row = [];
        for (var j = 0; j < GRID_SIZE; j++) {
            row[j] = grid = new Grid();
            board += "<td><img src='" + grid.image() +
                     "' id='" + i + "-" + j + "' class='cross'></td>";
        }
        board += "</tr>"
    }
    board += "</table>";
    document.getElementById("board").innerHTML = board;

    var play = function (i, j, rewind) {
        var grid = state[i][j];
        grid.play(rewind ? 0 : player);
        player = other(player);
        document.getElementById(i + '-' + j).src = grid.image();
    }
    var crosses = document.getElementsByClassName("cross");
    for (var k = 0; k < crosses.length; k++) {
        crosses[k].addEventListener("click", function () {
            var location = this.id.split("-");
            var i = parseInt(location[0]), j = parseInt(location[1]);
            grid = state[i][j];
            if (grid.isEmpty()) {
                history.push([i, j]);
                play(i, j);
                if (rules.win(i, j, state)) {
                    alert(name(other(player)) + " wins!");
                }
            }
        });
    }

    document.getElementById("rewind").addEventListener("click", function () {
        if (history.length > 0) {
            var location = history.pop();
            play(location[0], location[1], true);
        }
    });
})();
