package gomoku;

import java.util.Collections;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Random;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;

/** Advanced MinimaxAgent
 * @author TakLee96 */
public class MinimaxAgent extends Agent {
    /***********************
     *** CLASS CONSTANTS ***
     ***********************/
    // maximum evaluation score
    private static final double infinity = 1E10;
    // discount rate
    private static final double gamma = 0.99;
    // maximum search depth
    private static final int maxDepth = 20;
    // maximum big search depth
    private static final int maxBigDepth = 5;
    // separate big depth and normal depth
    private static final int bigDepthThreshold = 5;
    // branching factor
    private static final int branch = 24;
    // the random
    private static final Random random = new Random();

    /**********************
     *** HELPER CLASSES ***
     **********************/
    private static class Node {
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

    private static Comparator<Node> blackComparator = new Comparator<Node>(){
        public int compare(Node a, Node b) { return (int) (b.v - a.v); }
    };
    private static Comparator<Node> whiteComparator = new Comparator<Node>(){
        public int compare(Node a, Node b) { return (int) (a.v - b.v); }
    };

    /***************************
     *** INSTANCE ATTRIBUTES ***
     ***************************/
    // weights for evaluating states where black plays next
    private Counter blackWeights;
    // weights for evaluating states where white plays next
    private Counter whiteWeights;
    // total time for thinking
    private long thinking;
    // a cache for storing responses and evaluation scores
    private HashMap<Set<Move>, Node> memo;

    /*******************
     *** CONSTRUCTOR ***
     *******************/
    public MinimaxAgent(boolean isBlack) {
        super(isBlack);
        blackWeights = new Counter();
        whiteWeights = new Counter();
        thinking = 0;
        Counter.read(blackWeights, whiteWeights);
        memo = new HashMap<Set<Move>, Node>();
    }

    /********************
     *** CORE UTILITY ***
     ********************/
    private double value(State s) {
        if (s.isBlacksTurn()) return blackWeights.mul(s.extractFeatures());
        return whiteWeights.mul(s.extractFeatures());
    }

    private Node maxvalue(State s, double alpha, double beta, int depth, int bigDepth, Set<Action> actions) {
        Action maxaction = null; double maxvalue = -infinity, val = 0;
        Rewinder rewinder = null;
        for (Action a : actions) {
            if (depth == 1) s.evaluate(a);
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
        Rewinder rewinder = null;
        for (Action a : actions) {
            if (depth == 1) s.evaluate(a);
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

        Set<Move> prev = s.previousMoves(depth);
        Node memoized = memo.get(prev);
        if (memoized != null) return memoized;

        Set<Action> actions = getActions(s);
        if (actions.size() == 0)
            return new Node(s.randomAction(), (s.isBlacksTurn()) ? -infinity : infinity);
        if (actions.size() > bigDepthThreshold)
            bigDepth += 1;

        if (depth == 0) {
            s.highlight(actions);
            if (actions.size() == 1)
                for (Action a : actions)
                    return new Node(a, 0);
        }

        depth += 1;
        Node node = null; boolean who = s.isBlacksTurn();
        if (who) node = maxvalue(s, alpha, beta, depth, bigDepth, actions);
        else     node = minvalue(s, alpha, beta, depth, bigDepth, actions);
        memo.put(prev, node);
        return node;
    }

    private int four(Counter features, boolean isBlack) {
        if (isBlack)
            return (features.getInt("-oooox") + features.getInt("four-o"));
        return (features.getInt("-xxxxo") + features.getInt("four-x"));
    }
    private int straightFour(Counter features, boolean isBlack) {
        if (isBlack)
            return features.getInt("-oooo-");
        return features.getInt("-xxxx-");
    }
    private int three(Counter features, boolean isBlack) {
        if (isBlack)
            return (features.getInt("-o-oo-") +
                    features.getInt("-oo-o-") +
                    features.getInt("-ooo-"));
        return (features.getInt("-x-xx-") +
                features.getInt("-xx-x-") +
                features.getInt("-xxx-"));
    }

    private Set<Action> movesExtendFour(State s, Counter features) {
        Set<Action> result = new HashSet<Action>(1, 2);
        String win = (s.isBlacksTurn()) ? "win-o" : "win-x";
        for (Action a : s.getLegalActions()) {
            if (Extractor.diffFeatures(s, a).getInt(win) > 0) {
                result.add(a);
                return result;
            }
        }
        throw new RuntimeException("my four is missing?");
    }
    private Set<Action> movesCounterFour(State s, Counter features) {
        Set<Action> result = new HashSet<Action>(1, 2);
        boolean w = s.isBlacksTurn();
        for (Action a : s.getLegalActions()) {
            if (-four(Extractor.diffFeatures(s, a), !w) >= four(features, !w)) {
                result.add(a);
                return result;
            }
        }
        return result;
    }
    private Set<Action> movesExtendThree(State s, Counter features) {
        Set<Action> result = new HashSet<Action>(1, 2);
        boolean w = s.isBlacksTurn();
        for (Action a : s.getLegalActions()) {
            if (straightFour(Extractor.diffFeatures(s, a), w) > 0) {
                result.add(a);
                return result;
            }
        }
        if (three(features, !w) > 0)
            return movesCounterThree(s, features);
        return movesBestGrowth(s, features);
    }
    private Set<Action> movesCounterThree(State s, Counter features) {
        Set<Action> result = new HashSet<Action>(3, 2);
        boolean w = s.isBlacksTurn(); Counter diff = null;
        for (Action a : s.getLegalActions()) {
            diff = Extractor.diffFeatures(s, a);
            if (-three(diff, !w) >= three(features, !w) ||
                four(diff, w) > 0)
                result.add(a);
        }
        return result;
    }
    private Set<Action> movesBestGrowth(State s, Counter features) {
        Action[] actions = s.getLegalActions();
        Set<Action> result = new HashSet<Action>();
        boolean w = s.isBlacksTurn();

        // nodes that looks contributive
        Node[] nodes = new Node[actions.length];
        for (int i = 0; i < actions.length; i++)
            nodes[i] = new Node(actions[i], heuristic(s, actions[i]));
        Arrays.sort(nodes, (w) ? blackComparator : whiteComparator);
        for (int i = 0; i < branch/3 && i < nodes.length; i++)
            result.add(nodes[i].a);

        // nodes that looks good to me
        nodes = new Node[actions.length]; Rewinder r = null;
        for (int i = 0; i < actions.length; i++) {
            r = s.move(actions[i]);
            nodes[i] = new Node(actions[i], value(s));
            s.rewind(r);
        }
        Arrays.sort(nodes, (w) ? blackComparator : whiteComparator);
        for (int i = 0; i < branch/3 && i < nodes.length; i++)
            result.add(nodes[i].a);

        // nodes that looks good to opponent
        nodes = new Node[actions.length];
        s.makeDangerousNullMove();
        for (int i = 0; i < actions.length; i++) {
            r = s.move(actions[i]);
            nodes[i] = new Node(actions[i], value(s));
            s.rewind(r);
        }
        s.rewindDangerousNullMove();
        Arrays.sort(nodes, (!w) ? blackComparator : whiteComparator);
        for (int i = 0; i < branch/3 && i < nodes.length; i++)
            result.add(nodes[i].a);

        // nodes that are randomly selected
        int diff = branch - result.size();
        for (int i = 0; i < diff; i++)
            result.add(actions[random.nextInt(actions.length)]);

        return result;
    }

    private int heuristic(State s, Action a) {
        Counter diff = Extractor.diffFeatures(s, a);
        int score = 0;
        for (String key : diff.keySet())
            score += Extractor.sign(key) * diff.getInt(key);
        return score;
    }

    private Set<Action> getActions(State s) {
        Counter f = s.extractFeatures();
        boolean w = s.isBlacksTurn();
        if (four(f, w) > 0 || straightFour(f, w) > 0)
            return movesExtendFour(s, f);
        if (straightFour(f, !w) > 0)
            return new HashSet<Action>(1);
        if (four(f, !w) > 0)
            return movesCounterFour(s, f);
        if (three(f, w) > 0)
            return movesExtendThree(s, f);
        if (three(f, !w) > 0)
            return movesCounterThree(s, f);
        return movesBestGrowth(s, f);
    }

    @Override
    public Action getAction(State s) {
        if (!s.isTurn(isBlack))
            throw new RuntimeException("not my turn");
        long time = System.currentTimeMillis();
        Node retval = null;
        if (!s.started()) {
            retval = new Node(s.start, 0);
        } else {
            memo.clear();
            retval = value(s, -infinity, infinity, 0, 0);
        }

        // For debug purposes
        time = System.currentTimeMillis() - time;
        thinking += time;
        String elapsed = time + "";
        String value = (retval.v == 0) ? "None" : ("" + (long) retval.v);
        s.unhighlight();
        System.out.println("Done: " + ((isBlack) ? "BLACK" : "WHITE") + " " +
                         retval.a + " " + value + " [" + elapsed + "ms]");
        if (retval.v > infinity / 10)
            System.out.println("AI thinks BLACK is gonna win");
        else if (retval.v < -infinity / 10)
            System.out.println("AI thinks WHITE is gonna win");
        return retval.a;
    }

    /*********************
     *** DEBUG UTILITY ***
     *********************/
    public long thinking() { return thinking; }

    public static void main(String[] args) {
        GUIDriver.init();
        MinimaxAgent a = new MinimaxAgent(true);
        MinimaxAgent b = new MinimaxAgent(false);
        State s = new State();
        System.out.println("### BEGIN ###");
        while (!s.ended()) {
            if (s.isBlacksTurn()) s.move(a.getAction(s));
            else                  s.move(b.getAction(s));
            GUIDriver.drawBoard(s);
        }
        if (s.win(true))       System.out.println("### BLACK WINS ###");
        else if (s.win(false)) System.out.println("### WHITE WINS ###");
        else System.out.println("### TIE ###");
        System.out.println(s);
        System.out.println("Black average thinking time: " + (a.thinking() / s.blackMoves()) + "ms");
        System.out.println("White average thinking time: " + (b.thinking() / s.whiteMoves()) + "ms");
        System.out.println("History: " + s.history());
        System.out.println("Close the window or press CTRL+C to terminate");
    }

}
