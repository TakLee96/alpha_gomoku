package gomoku;

/** Move, essentially just (x, y, w)
 * @author TakLee96 */
class Move extends Action {
    private boolean isBlack;
    public Move(int x, int y, boolean isBlack) {
        super(x, y);
        this.isBlack = isBlack;
    }

    public Move(Action a, boolean isBlack) {
        super(a);
        this.isBlack = isBlack;
    }

    @Override
    public boolean equals(Object other) {
        if (other == null) {
            return false;
        }
        Move o = (Move) other;
        return o.x == x && o.y == y && o.isBlack == isBlack;
    }

    @Override
    public String toString() {
        String xc, yc; int x = x(), y = y();
        if (x < 10) xc = x + "";
        else xc = ((char) (x - 10 + 'A')) + "";
        if (y < 10) yc = y + "";
        else yc = ((char) (y - 10 + 'A')) + "";
        String me = (isBlack) ? "o" : "x";
        return "Move(" + xc + ", " + yc + ", " + me + ")";
    }

    @Override
    public int hashCode() {
        return ((isBlack) ? 1 : -1) * (x * State.N + y);
    }
}
