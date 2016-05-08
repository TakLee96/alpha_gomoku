package gomoku;

import java.util.LinkedList;
import java.util.ArrayList;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

/** Advanced MinimaxAgent
 * @author TakLee96 */
public class MinimaxAgent extends ReflexAgent {

    private static final int maxDepth = 8;
    private static final int maxBigDepth = 4;

    private class Node {
        public Action a;
        public double v;
        public Node(Action a, double v) {
            this.a = a;
            this.v = v;
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
            val = gamma * value(s, alpha, beta, depth, bigDepth);
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
            val = gamma * value(s, alpha, beta, depth, bigDepth);
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
        if (s.wins(true))
            return new Node(null, infinity);
        if (s.wins(false))
            return new Node(null, -infinity);
        if (s.ended())
            return new Node(null, 0.0);
        if (depth == maxDepth || bigDepth == maxBigDepth)
            return new Node(null, value(s));
        depth += 1;
        Set<Action> actions = getActions(s);
        if (actions.size() == 0)
            return new Node(s.randomAction(), (s.isBlacksTurn()) ? -infinity : infinity);
        if (actions.size() > 1)
            bigDepth += 1;
        if (s.isBlacksTurn())
            return maxvalue(s, alpha, beta, depth, bigDepth, actions);
        return minvalue(s, alpha, beta, depth, bigDepth, actions);
    }

    // TODO:
    private boolean containsMyFour(Map<String, Integer> features) {}
    private boolean containsEnemyStraightFour(Map<String, Integer> features) {}
    private boolean containsEnemyFour(Map<String, Integer> features) {}
    private boolean containsEnemyThree(Map<String, Integer> features) {}

    private Set<Action> movesWinGame(State s, Map<String, Integer> features) {}
    private Set<Action> movesCounterEnemyFour(State s, Map<String, Integer> features) {}
    private Set<Action> movesCounterEnemyThree(State s, Map<String, Integer> features) {}
    private Set<Action> movesBestGrowth(State s, Map<String, Integer> features) {}

    private Set<Action> getActions(State s) {
        Map<String, Integer> features = s.extractFeatures();
        if containsMyFour(features)
            return movesWinGame(s, features);
        if containsEnemyStraightFour(features)
            return new HashSet<Action>(1);
        if containsEnemyFour(features)
            return movesCounterEnemyFour(s, features);
        if containsEnemyThree(features)
            return movesCounterEnemyThree(s, features);
        return movesBestGrowth(s, features);
    }

    @Override
    public Action getAction(State s) {
        if (!s.isTurn(isBlack))
            throw new RuntimeException("not my turn");
        return value(s, -infinity, infinity, 0, 0);
    }

}
