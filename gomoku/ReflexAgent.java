package gomoku;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;
import java.util.Map;

/** Basic ReflexAgent
 * @author TakLee96 */
public class ReflexAgent extends Agent {
    protected static final double infinity = 1E15;
    protected static final double gamma = 0.99;
    private static final Random random = new Random();

    protected Counter weights;
    public ReflexAgent(boolean isBlack) {
        super(isBlack);
        weights = new Counter("gomoku/weight.csv");
    }

    protected ReflexAgent(boolean isBlack, Counter w) {
        super(isBlack);
        weights = w;
    }

    protected double value(State s) {
        Map<String, Integer> features = s.extractFeatures();
        return value(s, features);
    }

    protected double value(State s, Map<String, Integer> features) {
        double total = 0.0;
        for (String key : s.extractFeatures().keySet())
            total += features.get(key) * weights.get(key);
        return total;
    }

    protected double nextMinValue(State s) {
        double minVal = infinity; double val = 0.0;
        LinkedList<Action> rewinder = null;
        for (Action a : s.getLegalActions()) {
            rewinder = s.move(a);
            val = value(s);
            if (val < minVal)
                minVal = val;
            s.rewind(rewinder);
        }
        if (minVal == infinity)
            throw new RuntimeException("everybody is huge!");
        return minVal;
    }

    protected double nextMaxValue(State s) {
        double maxVal = -infinity; double val = 0.0;
        LinkedList<Action> rewinder = null;
        for (Action a : s.getLegalActions()) {
            rewinder = s.move(a);
            val = value(s);
            if (val > maxVal)
                maxVal = val;
            s.rewind(rewinder);
        }
        if (maxVal == -infinity)
            throw new RuntimeException("everybody is small!");
        return maxVal;
    }

    protected Action nextMinAction(State s) {
        double minVal = infinity; double val = 0.0;
        ArrayList<Action> actions = new ArrayList<Action>();
        LinkedList<Action> rewinder = null;
        for (Action a : s.getLegalActions()) {
            rewinder = s.move(a);
            val = value(s);
            if (val < minVal) {
                minVal = val;
                actions.clear();
                actions.add(a);
            } else if (val == minVal) {
                actions.add(a);
            }
            s.rewind(rewinder);
        }
        if (minVal == infinity)
            throw new RuntimeException("everybody is huge!");
        if (actions.size() == 1) return actions.get(0);
        return actions.get(random.nextInt(actions.size()));
    }

    protected Action nextMaxAction(State s) {
        double maxVal = -infinity; double val = 0.0;
        ArrayList<Action> actions = new ArrayList<Action>();
        LinkedList<Action> rewinder = null;
        for (Action a : s.getLegalActions()) {
            rewinder = s.move(a);
            val = value(s);
            if (val > maxVal) {
                maxVal = val;
                actions.clear();
                actions.add(a);
            } else if (val == maxVal) {
                actions.add(a);
            }
            s.rewind(rewinder);
        }
        if (maxVal == -infinity)
            throw new RuntimeException("everybody is small!");
        if (actions.size() == 1) return actions.get(0);
        return actions.get(random.nextInt(actions.size()));
    }

    protected double nextValue(State s) {
        if (s.isBlacksTurn())
            return nextMaxValue(s);
        return nextMinValue(s);
    }

    protected Action getPolicy(State s) {
        if (!s.isTurn(isBlack))
            throw new RuntimeException("not my turn");
        if (!s.started())
            return s.start;
        if (isBlack) return nextMaxAction(s);
        return nextMinAction(s);
    }

    @Override
    public Action getAction(State s) {
        return getPolicy(s);
    }


}
