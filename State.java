import java.util.LinkedList;
import java.util.HashSet;
import java.util.Random;
import java.util.Map;

public class State {
    /* class constants */
    public static final int N = 15;
    public static final Action start = new Action(N/2, N/2);
    private static final Action[] neighbors = new Action[]{
        new Action(1, 0), new Action(-1, 0), new Action(0, 1), new Action(0, -1),
        new Action(1, 1), new Action(-1, 1), new Action(1, -1), new Action(-1, -1),
        new Action(2, 0), new Action(-2, 0), new Action(0, 2),  new Action(0, -2),
        new Action(2, 2),  new Action(-2, 2), new Action(2, -2),  new Action(-2, -2)
    };

    /* instance attributes & constructor */
    public int newX, newY;
    public String message;
    public LinkedList<Action> five;
    public LinkedList<Action> history;
    private int dx, dy;
    private boolean wins;
    private short numMoves;
    private Grid[][] board;
    private Random random;
    private HashSet<Action> legalActions;
    public State() {
        newX = -1; newY = -1;
        dx = 0; dy = 0;
        message = "It's Blue Circle's Turn.";
        wins = false;
        numMoves = 0;
        board = new Grid[N][N];
        five = new LinkedList<Action>();
        history = new LinkedList<Action>();
        legalActions = new HashSet<Action>();
        random = new Random(System.nanoTime());
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                board[i][j] = new Grid();
                legalActions.add(new Action(i, j));
            }
        }
    }

    public boolean started() {
        return numMoves > 0;
    }

    public boolean isBlacksTurn() {
        return numMoves % 2 == 0;
    }

    public boolean canMove(Action a) {
        return a != null && canMove(a.x(), a.y());
    }

    public boolean canMove(int x, int y) {
        return !end() && inBound(x, y) && board[x][y].isEmpty();
    }

    public void move(Action a) {
        move(a.x(), a.y());
    }

    private String who() {
        return (isBlacksTurn()) ? "Blue Circle" : "Green Cross";
    }

    public void move(int x, int y) {
        newX = x; newY = y;
        board[x][y].put(isBlacksTurn());
        Action move = new Action(x, y);
        legalActions.remove(move);
        history.add(move);
        wins = win(isBlacksTurn());
        if (wins) {
            message = who() + " wins the game!";
            numMoves++;
            boolean b = board[x][y].isBlack();
            x += dx; y += dy;
            while (inBound(x, y) && board[x][y].is(b)) {
                five.add(new Action(x, y));
                x = x + dx; y = y + dy;
            }
            x = newX - dx; y = newY - dy;
            while (inBound(x, y) && board[x][y].is(b)) {
                five.add(new Action(x, y));
                x = x - dx; y = y - dy;
            }
            five.add(new Action(newX, newY));
            newX = -1; newY = -1;
        } else {
            numMoves++;
            message = "It's " + who() + "'s Turn.";
        }
    }

    public Grid get(Action a) {
        return get(a.x(), a.y());
    }

    public Grid get(int x, int y) {
        return board[x][y];
    }

    public boolean end() {
        return wins || numMoves == N * N;
    }

    public boolean inBound(Action a) {
        return inBound(a.x(), a.y());
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

    public boolean blackWins() {
        return isBlacksTurn() && wins;
    }

    public boolean whiteWins() {
        return !isBlacksTurn() && wins;
    }

    public boolean win(boolean isBlack) {
        if (numMoves == 0 || isBlacksTurn() != isBlack) {
            return false;
        }
        if (1 + count(isBlack, newX, newY, (int) 1, (int) 0)
              + count(isBlack, newX, newY, (int)-1, (int) 0) >= 5) {
            dx = 1; dy = 0;
            return true;
        }
        if (1 + count(isBlack, newX, newY, (int) 0, (int) 1)
              + count(isBlack, newX, newY, (int) 0, (int)-1) >= 5) {
            dx = 0; dy = 1;
            return true;
        }
        if (1 + count(isBlack, newX, newY, (int) 1, (int) 1)
              + count(isBlack, newX, newY, (int)-1, (int)-1) >= 5) {
            dx = 1; dy = 1;
            return true;
        }
        if (1 + count(isBlack, newX, newY, (int) 1, (int)-1)
              + count(isBlack, newX, newY, (int)-1, (int) 1) >= 5) {
            dx = 1; dy = -1;
            return true;
        }
        return false;
    }

    public Action[] getLegalActions() {
        return legalActions.toArray(new Action[legalActions.size()]);
    }

    public Action randomAction() {
        return getLegalActions()[random.nextInt(legalActions.size())];
    }

    public void rewind() {
        Action last = history.pollLast();
        board[last.x()][last.y()].clean();
        legalActions.add(last);
        if (wins) {
            wins = false;
            five.clear();
        }
        numMoves--;
        message = "It's " + who() + "'s Turn.";
        if (history.isEmpty()) {
            newX = -1; newY = -1;
        } else {
            Action temp = history.getLast();
            newX = temp.x(); newY = temp.y();
        }
    }

    public Map<String, Integer> extractFeatures() {
        return Extractor.extractFeatures(this);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("- ");
        for (int k = 0; k < N; k++) {
            if (k < 10) {
                sb.append(k);
            } else {
                sb.append((char) (k - 10 + 'A'));
            }
            sb.append(" ");
        }
        sb.append("Y\n");
        for (int i = 0; i < N; i++) {
            if (i < 10) {
                sb.append(i);
            } else {
                sb.append((char) (i - 10 + 'A'));
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
        sb.append("X ");
        for (int l = 0; l <= N; l++) {
            sb.append("- ");
        }
        sb.append("\n");
        return sb.toString();
    }

}
