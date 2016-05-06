package gomoku;

/** Action, essentially just (x, y)
 * @author TakLee96 */
public class Action {
    private int x, y;
    public int x() { return x; }
    public int y() { return y; }
    public Action(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public Action(Action a) {
        this.x = a.x;
        this.y = a.y;
    }

    @Override
    public boolean equals(Object other) {
        if (other == null) {
            return false;
        }
        Action o = (Action) other;
        return o.x == x && o.y == y;
    }

    @Override
    public String toString() {
        String xc, yc;
        if (x < 10) xc = x + "";
        else xc = ((char) (x - 10 + 'A')) + "";
        if (y < 10) yc = y + "";
        else yc = ((char) (y - 10 + 'A')) + "";
        return "(" + xc + "," + yc + ")";
    }

    @Override
    public int hashCode() {
        return x * 1009 + y;
    }
}
