package gomoku;

import java.util.regex.Pattern;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** feature Extractor for gomoku State
 * @author TakLee96 */
public class Extractor {

    private static final Action[] directions = new Action[]{
        new Action(1, 0), new Action(0, 1), new Action(1, 1), new Action(-1, 1)
    };

    private static final HashSet<String> fundamentalFeatures =
        new HashSet<String>(Arrays.asList(new String[]{
        "-xxo", "-x-xo", "-oox", "-o-ox",
        "-x-x-", "-xx-", "-o-o-", "-oo-",
        "-x-xxo", "-xxxo", "-o-oox", "-ooox",
        "-x-xx-", "-xxx-", "-o-oo-", "-ooo-",
        "-xxxx-", "-xxxxo", "-oooo-", "-oooox"
    }));
    private static Pattern blackFour = Pattern.compile(".*[x-](o-ooo|oo-oo|ooo-o)[x-].*");
    private static Pattern whiteFour = Pattern.compile(".*[o-](x-xxx|xx-xx|xxx-x)[o-].*");
    private static Pattern jumpThree = Pattern.compile(".[xo][xo]-[xo].");
    private static String format(String feature) {
        if (feature.length() < 3)
            return null;
        if (feature.contains("oooooo") || feature.contains("xxxxxx"))
            return null;
        if (feature.contains("ooooo"))
            return "win-o";
        if (feature.contains("xxxxx"))
            return "win-x";
        if (blackFour.matcher(feature).matches())
            return "four-o";
        if (whiteFour.matcher(feature).matches())
            return "four-x";
        if (feature.length() > 6)
            return null;
        if (feature.charAt(0) != EMPTY.charAt(0) &&
            feature.charAt(feature.length()-1) == EMPTY.charAt(0))
            feature = new StringBuilder(feature).reverse().toString();
        if (jumpThree.matcher(feature).matches())
            feature = feature.charAt(0) +
                new StringBuilder(feature.substring(1, 5)).reverse().toString()
                + feature.charAt(feature.length()-1);
        if (fundamentalFeatures.contains(feature))
            return feature;
        return null;
    }

    private static final String BLACK = "o";
    private static final String WHITE = "x";
    private static final String EMPTY = "-";

    public static Counter diffFeatures(State s, Action a) {
        return diffFeatures(s, a.x(), a.y());
    }

    public static Counter diffFeatures(State s, int x, int y) {
        Counter count = new Counter();
        for (Action a : directions)
            check(s, x, y, a.x(), a.y(), count);
        return count;
    }

    public static int sign(String feature) {
        if (feature.equals("four-o") || feature.equals("win-o") ||
            (feature.charAt(0) == BLACK.charAt(0) && feature.charAt(1) == EMPTY.charAt(0)) ||
            (feature.charAt(0) == EMPTY.charAt(0) && feature.charAt(1) == BLACK.charAt(0)))
            return 1;
        return -1;
    }

    private static void check(State s, int x, int y, int dx, int dy, Counter count) {
        int posX = x + dx, posY = y + dy, negX = x - dx, negY = y - dy;
        boolean who = s.isBlacksTurn();
        if (!s.inBound(posX, posY) && !s.inBound(negX, negY))
            // both out of bound; imagine at top-left with direction top-right
            return;
        if (!s.inBound(posX, posY))
            // right out of bound
            checkOnly(s, x, y, -dx, -dy, count);
        else if (!s.inBound(negX, negY))
            // left out of bound
            checkOnly(s, x, y, dx, dy, count);
        else if (s.get(posX, posY).is(who) && s.get(negX, negY).is(who))
            // left right all me
            checkConnection(s, x, y, dx, dy, count);
        else if (s.get(posX, posY).is(!who) && s.get(negX, negY).is(!who))
            // left right all enemy
            checkDisconnection(s, x, y, dx, dy, count);
        else if ((s.get(posX, posY).is(!who) && s.get(negX, negY).is(who)) ||
                 (s.get(posX, posY).is(who)  && s.get(negX, negY).is(!who))) {
            // one side enemy other side me
            checkOnly(s, x, y, dx, dy, count);
            checkOnly(s, x, y, -dx, -dy, count); }
        else if (s.get(posX, posY).isEmpty() && s.get(negX, negY).isEmpty())
            // both sides empty
            checkOneSide(s, x, y, dx, dy, count);
        else
            // one side empty other side something
            checkBothSide(s, x, y, dx, dy, count);
    }

    private static void checkOneSide(State s, int x, int y, int dx, int dy, Counter count) {
        // *o-(o)-o* or *o-(o)--* or *o-(o)-x*
        boolean who = s.isBlacksTurn();
        int llx = x - dx - dx, lly = y - dy - dy;
        int rrx = x + dx + dx, rry = y + dy + dy;
        Grid leftleft = (s.inBound(llx, lly)) ? s.get(llx, lly) : new Grid();
        Grid rightright = (s.inBound(rrx, rry)) ? s.get(rrx, rry) : new Grid();
        String ONE = (who) ? BLACK : WHITE;
        if (leftleft.is(who)) {
            if (rightright.is(who)) {
                // o-o-o
                if ((!s.inBound(llx - dx, lly - dy) || !s.get(llx - dx, lly - dy).is(who)) &&
                    (!s.inBound(rrx + dx, rry + dy) || !s.get(rrx + dx, rry + dy).is(who))) {
                    if (who) count.put("o-o-o", count.getInt("o-o-o") + 1);
                    else     count.put("x-x-x", count.getInt("x-x-x") + 1);
                } else {
                    String left = leftHelper(s, llx + dx, lly + dy, dx, dy);
                    String right = rightHelper(s, rrx - dx, rry - dy, dx, dy);
                    String newfeature = format(left + EMPTY + ONE + EMPTY + right);
                    count.put(newfeature, count.getInt(newfeature) + 1);
                }
            } else {
                // o-o--
                String left = leftHelper(s, llx + dx, lly + dy, dx, dy);
                String oldfeature = format(left + EMPTY);
                String newfeature = format(left + EMPTY + ONE + EMPTY);
                count.put(oldfeature, count.getInt(oldfeature) - 1);
                count.put(newfeature, count.getInt(newfeature) + 1);
            }
        } else if (rightright.is(who)) {
            // --o-o
            String right = rightHelper(s, rrx - dx, rry - dy, dx, dy);
            String oldfeature = format(EMPTY + right);
            String newfeature = format(EMPTY + ONE + EMPTY + right);
            count.put(oldfeature, count.getInt(oldfeature) - 1);
            count.put(newfeature, count.getInt(newfeature) + 1);
        }
    }

    private static void checkBothSide(State s, int x, int y, int dx, int dy, Counter count) {
        // *oo(o)-* or *o(o)-o* or *xx(o)-* or *x(o)-o*
        if (s.get(x + dx, y + dy).isEmpty()) {
            checkBothSide(s, x, y, -dx, -dy, count);
            return;
        }
        // something on my right
        boolean who = s.isBlacksTurn();
        String ONE = (who) ? BLACK : WHITE;
        String OTHER = (who) ? WHITE : BLACK;
        if (s.get(x + dx, y + dy).is(who)) {
            // one-way or two-way consective
            int llx = x - dx - dx, lly = y - dy - dy;
            Grid leftleft = (s.inBound(llx, lly)) ? s.get(llx, lly) : new Grid();
            if (leftleft.is(who)) {
                String left = leftHelper(s, llx + dx, lly + dy, dx, dy);
                String right = rightHelper(s, x, y, dx, dy);
                String newfeature = format(left + EMPTY + ONE + right);
                count.put(newfeature, count.getInt(newfeature) + 1);
            } else {
                String right = rightHelper(s, x, y, dx, dy);
                String oldfeature = format(EMPTY + right);
                String newfeature = format(EMPTY + ONE + right);
                count.put(oldfeature, count.getInt(oldfeature) - 1);
                count.put(newfeature, count.getInt(newfeature) + 1);
            }
        } else {
            // one-way blocked consective and/or blocked enemy
            int llx = x - dx - dx, lly = y - dy - dy;
            Grid leftleft = (s.inBound(llx, lly)) ? s.get(llx, lly) : new Grid();
            if (leftleft.is(who)) {
                String left = leftHelper(s, llx + dx, lly + dy, dx, dy);
                String right = rightHelper(s, x, y, dx, dy);
                String oldfeature = format(EMPTY + right);
                String newfeature1 = format(ONE + right);
                String newfeature2 = format(left + EMPTY + ONE + OTHER);
                count.put(oldfeature, count.getInt(oldfeature) - 1);
                count.put(newfeature1, count.getInt(newfeature1) + 1);
                count.put(newfeature2, count.getInt(newfeature2) + 1);
            } else {
                String right = rightHelper(s, x, y, dx, dy);
                String oldfeature = format(EMPTY + right);
                String newfeature = format(ONE + right);
                count.put(oldfeature, count.getInt(oldfeature) - 1);
                count.put(newfeature, count.getInt(newfeature) + 1);
            }
        }
    }

    private static String leftHelper(State s, int x, int y, int dx, int dy) {
        int nx = x - dx; int ny = y - dy;
        String ONE = null, OTHER = null;
        boolean who = s.get(nx, ny).isBlack();
        if (who) {
            ONE = BLACK; OTHER = WHITE; who = true;
        } else {
            ONE = WHITE; OTHER = BLACK; who = false;
        }
        String feature = "";
        while (s.inBound(nx, ny)) {
            Grid next = s.get(nx, ny);
            if (next.is(who)) {
                feature = ONE + feature;
                nx -= dx; ny -= dy;
            } else if (next.isEmpty() && s.inBound(nx - dx, ny - dy) &&
                s.get(nx - dx, ny - dy).is(who)) {
                feature = EMPTY + feature;
                nx -= dx; ny -= dy;
            } else {
                if (next.isEmpty())
                    feature = EMPTY + feature;
                else
                    feature = OTHER + feature;
                break;
            }
        }
        if (!s.inBound(nx, ny))
            feature = OTHER + feature;
        return feature;
    }

    private static String rightHelper(State s, int x, int y, int dx, int dy) {
        int nx = x + dx; int ny = y + dy;
        String ONE = null, OTHER = null;
        boolean who = s.get(nx, ny).isBlack();
        if (who) {
            ONE = BLACK; OTHER = WHITE; who = true;
        } else {
            ONE = WHITE; OTHER = BLACK; who = false;
        }
        String feature = "";
        while (s.inBound(nx, ny)) {
            Grid next = s.get(nx, ny);
            if (next.is(who)) {
                feature = feature + ONE;
                nx += dx; ny += dy;
            } else if (next.isEmpty() && s.inBound(nx + dx, ny + dy) &&
                s.get(nx + dx, ny + dy).is(who)) {
                feature = feature + EMPTY;
                nx += dx; ny += dy;
            } else {
                if (next.isEmpty())
                    feature = feature + EMPTY;
                else
                    feature = feature + OTHER;
                break;
            }
        }
        if (!s.inBound(nx, ny))
            feature = feature + OTHER;
        return feature;
    }

    private static void checkConnection(State s, int x, int y, int dx, int dy, Counter count) {
        String left = leftHelper(s, x, y, dx, dy);
        String right = rightHelper(s, x, y, dx, dy);
        String oldfeature = format(left + EMPTY + right);
        String newfeature = format(left + ((s.isBlacksTurn()) ? BLACK : WHITE) + right);
        count.put(oldfeature, count.getInt(oldfeature) - 1);
        count.put(newfeature, count.getInt(newfeature) + 1);
    }

    private static void checkDisconnection(State s, int x, int y, int dx, int dy, Counter count) {
        String left = leftHelper(s, x, y, dx, dy);
        String right = rightHelper(s, x, y, dx, dy);
        String feature = format(left + EMPTY + right);
        count.put(feature, count.getInt(feature) - 1);
        String newLeft = format(left + ((s.isBlacksTurn()) ? BLACK : WHITE));
        String newRight = format(((s.isBlacksTurn()) ? BLACK : WHITE) + right);
        count.put(newLeft, count.getInt(newLeft) + 1);
        count.put(newRight, count.getInt(newRight) + 1);
    }

    private static void checkOnly(State s, int x, int y, int dx, int dy, Counter count) {
        // assume left out of bound, want to check right
        int nx = x + dx, ny = y + dy;
        boolean who = s.isBlacksTurn();
        String right = null, oldfeature = null, newfeature = null;
        if (s.inBound(nx, ny) && !s.get(nx, ny).isEmpty()) {
            right = rightHelper(s, x, y, dx, dy);
        } else if (s.inBound(nx + dx, ny + dy) && s.get(nx + dx, ny + dy).is(who)) {
            right = EMPTY + rightHelper(s, nx, ny, dx, dy);
            nx += dx; ny += dy;
        } else {
            return;
        }
        if (who) {
            if (s.get(nx, ny).is(true)) {
                oldfeature = EMPTY + right;
                newfeature = WHITE + BLACK + right;
            } else {
                oldfeature = EMPTY + right;
                newfeature = BLACK + right;
            }
        } else {
            if (s.get(nx, ny).is(true)) {
                oldfeature = EMPTY + right;
                newfeature = WHITE + right;
            } else {
                oldfeature = EMPTY + right;
                newfeature = BLACK + WHITE + right;
            }
        }
        oldfeature = format(oldfeature);
        newfeature = format(newfeature);
        count.put(oldfeature, count.getInt(oldfeature) - 1);
        count.put(newfeature, count.getInt(newfeature) + 1);
    }

}
