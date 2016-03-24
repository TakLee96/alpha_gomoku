(function () {
    var player = 1; // 0 for empty, 1 for blue and 2 for green
    var other = function (player) { return 3 - player };
    var human  = other(parseInt(document.getElementById("first").value));
    var name = function (player) { return (player == 1) ? 'blue' : 'green';  }
    var request = function (method, url, cb) {
        var xhr = new XMLHttpRequest();
        xhr.onreadystatechange = function () {
            if (xhr.readyState == 4) {
                if (xhr.status == 200) {
                    return cb && cb(null, xhr.responseText && JSON.parse(xhr.responseText));
                } else {
                    return cb && cb(new Error(method + " " + url + " failed with " + xhr.status));
                }
            }
        };
        xhr.ontimeout = function () {
            return cb && cb(new Error(method + " " + url + " timed out"));
        };
        xhr.open(method, url, true);
        xhr.send();
    }

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

    var state = [], row = null, grid = null;
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

    var won = false;
    var play = function (i, j) {
        var grid = state[i][j];
        grid.play(player);
        player = other(player);
        document.getElementById(i + '-' + j).src = grid.image();
    }
    var rewind = function (i, j) {
        var grid = state[i][j];
        grid.play(0);
        player = other(player);
        document.getElementById(i + '-' + j).src = grid.image();
    }
    var crosses = document.getElementsByClassName("cross");
    for (var k = 0; k < crosses.length; k++) {
        crosses[k].addEventListener("click", function () {
            var location = this.id.split("-");
            var i = parseInt(location[0]), j = parseInt(location[1]);
            grid = state[i][j];
            if (player == human && grid.isEmpty() && !won) {
                play(i, j);
                if (rules.win(i, j, state)) {
                    alert(name(other(player)) + " wins!");
                    won = true;
                } else {
                    request('POST', '/api?x='+i+'&y='+j, function (err, data) {
                        if (err) return alert(err);
                        var i = data['x'], j = data['y'];
                        play(i, j);
                        if (rules.win(i, j, state)) {
                            alert(name(other(player)) + " wins!");
                            won = true;
                        }
                    });
                }
            }
        });
    }

    document.getElementById("go").addEventListener("click", function () {
        request('POST', '/api?new=1&first=' + document.getElementById("first").value, function (err, data) {
            if (err) return alert(err);
            human = other(parseInt(document.getElementById("first").value));
            var i = data['x'], j = data['y'];
            if (i != undefined && j != undefined) {
                play(i, j);    
            }
            document.getElementById("board").className = "";
            document.getElementById("options").className = "hidden";
        });
    });

    document.getElementById("load").addEventListener("click", function () {
        document.getElementById("board").className = "";
        document.getElementById("options").className = "hidden";
        document.getElementById("control").className = "";
        var hist = JSON.parse(prompt("Please enter the history JSON object").replace(/\(/g, "[").replace(/\)/g, "]"));
        var i = 0;
        document.getElementById("next").addEventListener("click", function () {
            move = hist[i++];
            document.getElementById("next").disabled = i == hist.length;
            document.getElementById("prev").disabled = i == 0;
            play(move[0], move[1]);
        });
        document.getElementById("prev").addEventListener("click", function () { 
            move = hist[--i];
            document.getElementById("prev").disabled = i == 0;
            document.getElementById("next").disabled = i == hist.length;
            rewind(move[0], move[1]);
        });
    });

    window.onbeforeunload = function () {
        request('POST', 'api?end=1', console.log);
    };
})();
