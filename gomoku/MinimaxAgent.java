package gomoku;

import java.util.PriorityQueue;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Comparator;
import java.util.Map;
import java.util.Set;

/** Advanced MinimaxAgent
 * @author TakLee96 */
public class MinimaxAgent extends ReflexAgent {

    private static final int maxDepth = 10;
    private static final int maxBigDepth = 5;
    private static final int branch = 10;

    private class Node {
        public Action a;
        public double v;
        public Node(Action a, double v) {
            this.a = a;
            this.v = v;
        }
        @Override
        public boolean equals(Object other) {
            if (other == null) return false;
            Node o = (Node) other;
            return a.equals(o.a);
        }
        @Override
        public int hashCode() {
            return a.hashCode();
        }
        @Override
        public String toString() {
            return "Node(" + a.toString() + ", " + v + ")";
        }
    }

    private class NodeComparator implements Comparator<Node> {
        boolean isBlack;
        public NodeComparator(boolean isBlack) {
            this.isBlack = isBlack;
        }
        @Override
        public int compare(Node a, Node b) {
            if (isBlack) return (int) (b.v - a.v);
            return (int) (a.v - b.v);
        }
    }

    public MinimaxAgent(boolean isBlack) {
        super(isBlack, new Counter("gomoku/myweight.csv"));
    }

    private Node maxvalue(State s, double alpha, double beta, int depth, int bigDepth, Set<Action> actions) {
        Action maxaction = null; double maxvalue = -infinity, val = 0;
        LinkedList<Action> rewinder = null;
        for (Action a : actions) {
            rewinder = s.move(a);
            val = gamma * value(s, alpha, beta, depth, bigDepth).v;
            s.rewind(rewinder);
            if (val > maxvalue) {
                maxvalue = val;
                maxaction = a;
            }
            if (val > beta) {
                return new Node(maxaction, maxvalue);
            }
            alpha = Math.max(alpha, val);
        }
        if (maxaction == null) throw new RuntimeException("everybody is too small");
        return new Node(maxaction, maxvalue);
    }

    private Node minvalue(State s, double alpha, double beta, int depth, int bigDepth, Set<Action> actions) {
        Action minaction = null; double minvalue = infinity, val = 0;
        LinkedList<Action> rewinder = null;
        for (Action a : actions) {
            rewinder = s.move(a);
            val = gamma * value(s, alpha, beta, depth, bigDepth).v;
            s.rewind(rewinder);
            if (val < minvalue) {
                minvalue = val;
                minaction = a;
            }
            if (val < alpha) {
                return new Node(minaction, minvalue);
            }
            beta = Math.min(beta, val);
        }
        if (minaction == null) throw new RuntimeException("everybody is too large");
        return new Node(minaction, minvalue);
    }

    private Node value(State s, double alpha, double beta, int depth, int bigDepth) {
        if (s.win(true))
            return new Node(null, infinity);
        if (s.win(false))
            return new Node(null, -infinity);
        if (s.ended())
            return new Node(null, 0.0);
        if (depth == maxDepth || bigDepth == maxBigDepth)
            return new Node(null, value(s));
        depth += 1;
        Set<Action> actions = getActions(s);
        if (actions.size() == 0)
            return new Node(s.randomAction(), (s.isBlacksTurn()) ? -infinity : infinity);
        if (actions.size() > 3)
            bigDepth += 1;
        if (s.isBlacksTurn())
            return maxvalue(s, alpha, beta, depth, bigDepth, actions);
        return minvalue(s, alpha, beta, depth, bigDepth, actions);
    }

    private boolean containsFour(boolean next, Map<String, Integer> features, boolean isBlack) {
        String prefix = (next) ? "[black]" : "[white]";
        if (isBlack)
            return (features.containsKey(prefix + "-oooox") ||
                    features.containsKey(prefix + "four-o"));
        return (features.containsKey(prefix + "-xxxxo") ||
                features.containsKey(prefix + "four-x"));
    }
    private boolean containsStraightFour(boolean next, Map<String, Integer> features, boolean isBlack) {
        String prefix = (next) ? "[black]" : "[white]";
        if (isBlack)
            return features.containsKey(prefix + "-oooo-");
        return features.containsKey(prefix + "-xxxx-");
    }
    private boolean containsThree(boolean next, Map<String, Integer> features, boolean isBlack) {
        String prefix = (next) ? "[black]" : "[white]";
        if (isBlack)
            return (features.containsKey(prefix + "-o-oo-") ||
                    features.containsKey(prefix + "-oo-o-") ||
                    features.containsKey(prefix + "-ooo-"));
        return (features.containsKey(prefix + "-x-xx-") ||
                features.containsKey(prefix + "-xx-x-") ||
                features.containsKey(prefix + "-xxx-"));
    }

    private Set<Action> movesExtendFour(State s, Map<String, Integer> features) {
        Set<Action> result = new HashSet<Action>(1, 2);
        boolean w = s.isBlacksTurn();
        for (Action a : s.getLegalActions()) {
            LinkedList<Action> rewinder = s.move(a);
            if (s.win(w))
                result.add(a);
            s.rewind(rewinder);
            if (!result.isEmpty()) return result;
        }
        throw new RuntimeException("my four is missing?");
    }
    private Set<Action> movesCounterFour(State s, Map<String, Integer> features) {
        Set<Action> result = new HashSet<Action>(1, 2);
        boolean w = s.isBlacksTurn();
        for (Action a : s.getLegalActions()) {
            LinkedList<Action> rewinder = s.move(a);
            if (!containsFour(!w, s.extractFeatures(), !w))
                result.add(a);
            s.rewind(rewinder);
            if (!result.isEmpty()) return result;
        }
        System.out.println("multiple four!");
        return result;
    }
    private Set<Action> movesExtendThree(State s, Map<String, Integer> features) {
        Set<Action> result = new HashSet<Action>(1, 2);
        boolean w = s.isBlacksTurn();
        for (Action a : s.getLegalActions()) {
            LinkedList<Action> rewinder = s.move(a);
            if (containsStraightFour(!w, s.extractFeatures(), w))
                result.add(a);
            s.rewind(rewinder);
            if (!result.isEmpty()) return result;
        }
        throw new RuntimeException("my three is missing?");
    }
    private Set<Action> movesCounterThree(State s, Map<String, Integer> features) {
        Set<Action> result = new HashSet<Action>(3, 2);
        boolean w = s.isBlacksTurn();
        for (Action a : s.getLegalActions()) {
            LinkedList<Action> rewinder = s.move(a);
            if (!containsThree(!w, s.extractFeatures(), !w))
                result.add(a);
            s.rewind(rewinder);
        }
        return result;
    }
    private Set<Action> movesBestGrowth(State s, Map<String, Integer> features) {
        Set<Action> result = new HashSet<Action>();
        boolean w = s.isBlacksTurn();
        Action[] actions = s.getLegalActions();
        PriorityQueue<Node> moves = new PriorityQueue<Node>(
            actions.length, new NodeComparator(w));
        for (Action a : actions) {
            LinkedList<Action> rewinder = s.move(a);
            moves.add(new Node(a, value(s)));
            s.rewind(rewinder);
        }
        for (int i = 0; i < branch && !moves.isEmpty(); i++)
            result.add(moves.poll().a);
        return result;
    }

    private Set<Action> getActions(State s) {
        Map<String, Integer> f = s.extractFeatures();
        boolean w = s.isBlacksTurn();
        if (containsFour(w, f, w) || containsStraightFour(w, f, w))
            return movesExtendFour(s, f);
        if (containsStraightFour(w, f, !w))
            return new HashSet<Action>(1);
        if (containsFour(w, f, !w))
            return movesCounterFour(s, f);
        if (containsThree(w, f, w))
            return movesExtendThree(s, f);
        if (containsThree(w, f, !w))
            return movesCounterThree(s, f);
        return movesBestGrowth(s, f);
    }

    @Override
    public Action getAction(State s) {
        if (!s.isTurn(isBlack))
            throw new RuntimeException("not my turn");
        return value(s, -infinity, infinity, 0, 0).a;
    }

}
