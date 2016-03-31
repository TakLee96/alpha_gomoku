import java.util.ArrayList;

public class State {
    /* class constants */
    public static final int N = 15;

    /* instance attributes & constructor */
    public int newX, newY;
    public String message;
    public ArrayList<Tuple<Integer>> five;
    private int dx, dy;
    private boolean blackWins, whiteWins;
    private short numMoves;
    private Grid[][] board;
    public State() {
        newX = -1; newY = -1;
        dx = 0; dy = 0;
        message = "It's Blue Circle's Turn.";
        blackWins = false; whiteWins = false;
        numMoves = 0;
        board = new Grid[N][N];
        five = new ArrayList<Tuple<Integer>>(5);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                board[i][j] = new Grid();
            }
        }
    }

    private boolean isBlacksTurn() {
        return numMoves % 2 == 0;
    }

    public void move(int x, int y) {
        if (!end() && inBound(x, y) && board[x][y].isEmpty()) {
            newX = x; newY = y;
            board[x][y].put(isBlacksTurn());
            /*check if win and extract feature*/
            blackWins = win(true); whiteWins = win(false);
            numMoves++;
            if (blackWins || whiteWins) {
                String winner = (!isBlacksTurn()) ? "Blue Circle" : "Green Cross";
                message = winner + " wins the game!";
                boolean b = board[x][y].isBlack();
                x += dx; y += dy; 
                while (inBound(x, y) && board[x][y].is(b)) {
                    five.add(new Tuple<Integer>(x, y));
                    x = x + dx; y = y + dy;
                }
                x = newX - dx; y = newY - dy;
                while (inBound(x, y) && board[x][y].is(b)) {
                    five.add(new Tuple<Integer>(x, y));
                    x = x - dx; y = y - dy;
                }
                five.add(new Tuple<Integer>(newX, newY));
                newX = -1; newY = -1;
            } else {
                String who = (isBlacksTurn()) ? "Blue Circle" : "Green Cross";
                message = "It's " + who + "'s Turn.";
            }
            System.out.println(this);
        }
    }

    public Grid get(int x, int y) {
        return board[x][y];
    }

    public boolean end() {
        return blackWins || whiteWins || numMoves == N * N;
    }

    public boolean inBound(int x, int y) {
        return (x >= 0 && x < N && y >= 0 && y < N);
    }

    private int count(boolean isBlack, int x, int y, int dx, int dy) {
        int count = 0;
        x += dx; y += dy;
        while (inBound(x, y) && board[x][y].is(isBlack)) {
            count += 1;
            x += dx; y += dy;
        }
        return count;
    }

    public boolean win(boolean isBlack) {
        if (numMoves == 0 || isBlacksTurn() != isBlack) {
            return false;
        }
        if (1 + count(isBlack, newX, newY, (int) 1, (int) 0)
              + count(isBlack, newX, newY, (int)-1, (int) 0) == 5) {
            dx = 1; dy = 0;
            return true;
        }
        if (1 + count(isBlack, newX, newY, (int) 0, (int) 1)
              + count(isBlack, newX, newY, (int) 0, (int)-1) == 5) {
            dx = 0; dy = 1;
            return true;
        }
        if (1 + count(isBlack, newX, newY, (int) 1, (int) 1)
              + count(isBlack, newX, newY, (int)-1, (int)-1) == 5) {
            dx = 1; dy = 1;
            return true;
        }
        if (1 + count(isBlack, newX, newY, (int) 1, (int)-1)
              + count(isBlack, newX, newY, (int)-1, (int) 1) == 5) {
            dx = 1; dy = -1;
            return true;
        }
        return false;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("O ");
        for (int k = 1; k <= N; k++) {
            if (k < 10) {
                sb.append(k);
            } else {
                sb.append((char) (k - 10 + 'A'));
            }
            sb.append(" ");
        }
        sb.append("Y\n");
        for (int i = 0; i < N; i++) {
            if (i < 9) {
                sb.append(i + 1);
            } else {
                sb.append((char) (i - 9 + 'A'));
            }
            sb.append(" ");
            for (int j = 0; j < N; j++) {
                if (board[i][j].isEmpty()) {
                    sb.append("+ ");
                } else if (board[i][j].isBlack()) {
                    sb.append("o ");
                } else {
                    sb.append("x ");
                }
            }
            sb.append("|\n");
        }
        sb.append("X");
        for (int l = 0; l <= N; l++) {
            sb.append("- ");
        }
        return sb.toString();
    }

}