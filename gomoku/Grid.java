package gomoku;

/** Grid, recording black/white information
 * @author TakLee96 */
public class Grid {
    private static enum Who {
        EMPTY, BLACK, WHITE
    }

    private Who w;
    public Grid() {
        w = Who.EMPTY;
    }

    public boolean isBlack() {
        return w == Who.BLACK;
    }

    public boolean isWhite() {
        return w == Who.WHITE;
    }

    public boolean is(boolean isBlack) {
        if (isBlack) {
            return isBlack();
        }
        return isWhite();
    }

    public boolean isEmpty() {
        return w == Who.EMPTY;
    }

    public void put(boolean isBlack) {
        w = (isBlack) ? Who.BLACK : Who.WHITE;
    }

    public void clean() {
        w = Who.EMPTY;
    }

    public String getImage() {
        switch (w) {
            case BLACK: return "img/grid-blue.jpg";
            case WHITE: return "img/grid-green.jpg";
            default: return "img/grid-empty.jpg";
        }
    }

    @Override
    public String toString() {
        switch (w) {
            case BLACK: return "BLACK";
            case WHITE: return "WHITE";
            default: return "EMPTY";
        }
    }
}
