package gomoku;

/** Action, essentially just (x, y)
 * @author TakLee96 */
public class Action {
    protected int x, y;
    public int x() { return x; }
    public int y() { return y; }
    public Action(int x, int y) {
        this.x = x; this.y = y;
    }

    public Action(Action a) {
        this.x = a.x; this.y = a.y;
    }

    @Override
    public boolean equals(Object other) {
        if (other == null) {
            return false;
        }
        Action o = (Action) other;
        return x == o.x && y == o.y;
    }

    @Override
    public String toString() {
        String xc, yc; int x = x(), y = y();
        if (x < 10) xc = x + "";
        else xc = ((char) (x - 10 + 'A')) + "";
        if (y < 10) yc = y + "";
        else yc = ((char) (y - 10 + 'A')) + "";
        return "Action(" + xc + ", " + yc + ")";
    }

    @Override
    public int hashCode() {
        return x * State.N + y;
    }
}
