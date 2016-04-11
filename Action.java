public class Action {
    public int x;
    public int y;
    public Action(int x, int y) {
        this.x = x;
        this.y = y;
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
        return "(" + x + "," + y + ")";
    }

    @Override
    public int hashCode() {
        return x * 1009 + y;
    }
}