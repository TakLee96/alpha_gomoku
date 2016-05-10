package gomoku;

/** Action, essentially just (x, y)
 * @author TakLee96 */
public class Action {
    private int data;
    public int x() { return data % State.N; }
    public int y() { return data / State.N; }
    public Action(int x, int y) {
        this.data = y * State.N + x;
    }

    @Override
    public boolean equals(Object other) {
        if (other == null) {
            return false;
        }
        Action o = (Action) other;
        return o.data == data;
    }

    @Override
    public String toString() {
        String xc, yc; int x = x(), y = y();
        if (x < 10) xc = x + "";
        else xc = ((char) (x - 10 + 'A')) + "";
        if (y < 10) yc = y + "";
        else yc = ((char) (y - 10 + 'A')) + "";
        return "(" + xc + "," + yc + ")";
    }

    @Override
    public int hashCode() {
        return data;
    }
}
