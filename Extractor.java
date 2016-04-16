import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;

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
            // Reduce #Feature
            String reversed = null;
            if (feature.charAt(0) != EMPTY.charAt(0) && feature.length() < 7 &&
                feature.charAt(0) == feature.charAt(feature.length()-1)) return;
            else if (feature.contains("ooooo")) feature = "5o";
            else if (feature.contains("xxxxx")) feature = "5x";
            else if (feature.contains("-oooo-")) feature = "l4o";
            else if (feature.contains("-xxxx-")) feature = "l4x";
            else if (feature.contains("-oooo") ||
                     feature.contains("oooo-") ||
                     feature.contains("o-ooo") ||
                     feature.contains("ooo-o") ||
                     feature.contains("oo-oo")) feature = "d4o";
            else if (feature.contains("-xxxx") ||
                     feature.contains("xxxx-") ||
                     feature.contains("x-xxx") ||
                     feature.contains("x-xxx") ||
                     feature.contains("xx-xx")) feature = "d4x";
            else reversed = (new StringBuilder(feature)).reverse().toString();

            if (reversed != null && features.containsKey(reversed))
                features.put(reversed, features.get(reversed) + 1);
            else if (features.containsKey(feature))
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
