package gomoku;

import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;

/** feature Extractor for gomoku State
 * @author TakLee96 */
public class Extractor {

    private static enum Direction {
        LR(new Action(1, 0)), TB(new Action(0, 1)),
        TRBL(new Action(1, 1)), TLBR(new Action(-1, 1));
        private final Action a;
        private Direction(Action a) { this.a = a; }
        private Action val() { return this.a; }
    }
    private static final Direction[] directions = new Direction[]{
        Direction.LR, Direction.TB, Direction.TRBL, Direction.TLBR
    };

    private static final String BLACK = "o";
    private static final String WHITE = "x";
    private static final String EMPTY = "-";

    private static class Node {
        public Action a; public Direction d;
        public Node(Action a, Direction d) { this.a = a; this.d = d; }
        @Override
        public int hashCode() {
            return a.hashCode() + 1023 * d.val().hashCode();
        }
        @Override
        public boolean equals(Object other) {
            if (other == null) return false;
            Node n = (Node) other;
            return a.equals(n.a) && d == n.d;
        }
        @Override
        public String toString() {
            return "Node: " + a + " " + d.val();
        }
    }

    private static void check(State state, Action move, Direction direction,
        HashSet<Node> checked, HashMap<String, Integer> features) {
        int x = move.x(), y = move.y(),
            dx = direction.val().x(), dy = direction.val().y();
        int nx = x + dx, ny = y + dy;
        boolean who = state.get(x, y).isBlack();
        boolean jumped = false;
        String ONE = (who) ? BLACK : WHITE;
        String OTHER = (who) ? WHITE: BLACK;
        String feature = ONE;

        while (state.inBound(nx, ny)) {
            Grid next = state.get(nx, ny);
            if (next.is(who)) {
                feature = feature + ONE;
                checked.add(new Node(new Action(nx, ny), direction));
                nx += dx; ny += dy;
            } else if (next.isEmpty() && state.inBound(nx + dx, ny + dy) &&
                state.get(nx + dx, ny + dy).is(who) && !jumped) {
                jumped = true;
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
        if (!state.inBound(nx, ny))
            feature = feature + OTHER;

        nx = x - dx; ny = y - dy;
        jumped = false;
        while (state.inBound(nx, ny)) {
            Grid next = state.get(nx, ny);
            if (next.is(who)) {
                feature = ONE + feature;
                checked.add(new Node(new Action(nx, ny), direction));
                nx -= dx; ny -= dy;
            } else if (next.isEmpty() && state.inBound(nx - dx, ny - dy) &&
                state.get(nx - dx, ny - dy).is(who) && !jumped) {
                jumped = true;
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
        if (!state.inBound(nx, ny))
            feature = OTHER + feature;

        if (feature.length() > 3) {
            String reversed = null;
            if (feature.charAt(0) != EMPTY.charAt(0) && feature.length() < 7 &&
                feature.charAt(feature.length()-1) != EMPTY.charAt(0)) return;
            if (who) {
                if (feature.contains("oooooo")) return;
                if (feature.contains("ooooo")) feature = "win-o";
                else if (feature.matches("[x-]o-o-o-o[x-]")) feature = "jump-o";
                else if (feature.matches("[x-]o-ooo[x-]") ||
                         feature.matches("[x-]oo-oo[x-]") ||
                         feature.matches("[x-]ooo-o[x-]")) feature = "four-o";
                else if (feature.length() > 6) return;
                else if (feature.charAt(feature.length()-1) == EMPTY.charAt(0) &&
                         feature.charAt(0) != EMPTY.charAt(0))
                         feature = (new StringBuilder(feature)).reverse().toString();
            } else {
                if (feature.contains("xxxxxx")) return;
                if (feature.contains("xxxxx")) feature = "win-x";
                else if (feature.matches("[o-]x-x-x-x[o-]")) feature = "jump-x";
                else if (feature.matches("[o-]x-xxx[o-]") ||
                         feature.matches("[o-]xx-xx[o-]") ||
                         feature.matches("[o-]xxx-x[o-]")) feature = "four-x";
                else if (feature.length() > 6) return;
                else if (feature.charAt(feature.length()-1) == EMPTY.charAt(0) &&
                         feature.charAt(0) != EMPTY.charAt(0))
                         feature = (new StringBuilder(feature)).reverse().toString();
            }
            feature = ((state.isBlacksTurn()) ? "[black]" : "[white]") + feature;
            if (features.containsKey(feature))
                features.put(feature, features.get(feature) + 1);
            else features.put(feature, 1);
        }
    }

    public static Map<String, Integer> extractFeatures(State instance) {
        HashSet<Node> checked = new HashSet<Node>();
        HashMap<String, Integer> features = new HashMap<String, Integer>();
        for (Action move : instance.history) {
            for (Direction direction : directions) {
                Node n = new Node(move, direction);
                if (!checked.contains(n)) {
                    checked.add(n);
                    check(instance, move, direction, checked, features);
                }
            }
        }
        return features;
    }

}
