import java.util.Stack;
import java.util.Random;
import java.util.Map;

public class LearningAgent extends Agent {

    private static final double alpha = 0.1;
    private static final double gamma = 0.9;
    private static final double epsilon = 0.1;
    private static final double threshold = 0.015;
    private static final int numTraining = 1000;
    private static final int numTesting = 10;

    private Counter weights;
    private boolean doneTraining;
    private Random random;
    private Stack<Action> history;
    public LearningAgent(boolean isBlack) {
        super(isBlack);
        weights = new Counter();
        doneTraining = false;
        random = new Random(System.currentTimeMillis());
    }

    private double computeQValue(State s, Action a) {
        Map<String, Integer> features = s.extractFeatures(a);
        double val = 0.0;
        for (Map.Entry<String, Integer> e : features.entrySet()) {
            val += e.getValue() * weights.get(e.getKey());
        }
        return val;
    }

    private Action getPolicy(State s) {
        double bestVal = Double.MIN_VALUE;
        double val = 0.0; Action bestAction = null;
        for (Action a : s.getLegalActions()) {
            val = computeQValue(s, a);
            if (val > bestVal) {
                bestVal = val;
                bestAction = a;
            }
        }
        return bestAction;
    }

    private Action chooseAction(State s) {
        if (doneTraining || random.nextDouble() > epsilon) {
            return getPolicy(s);
        } else {
            return s.randomAction();
        }        
    }

    @Override
    public Action getAction(State s) {
        Action a = chooseAction(s);
        history.push(a);
        return a;
    }

    private void doneTraining() {
        doneTraining = true;
    }

    private void observeTransition(State s, Action a, double r, Counter newWeights) {
        Map<String, Integer> features = s.extractFeatures(a);
        double difference; double maxval; double val;
        for (Map.Entry<String, Integer> e : features.entrySet()) {
            difference = r - computeQValue(s, a);
            maxval = Double.MIN_VALUE;
            s.move(a);
            for (Action ap : s.getLegalActions()) {
                /* TODO:
                 * It is now enemy's turn, how can you ensure 
                 * that this Q value is valid/correct?
                 * This is the core difference between minimax q learning
                 * and other type such as normal q learning and opponent
                 * modeling q learning and prioritized sweeping
                 * UPDATE: for now, I will use normal q learning
                 */
                val = computeQValue(s, ap);
                if (val > maxval) {
                    maxval = val;
                }
            }
            s.rewindTill(a);
            difference += gamma * maxval;
            newWeights.put(e.getKey(), weights.get(e.getKey()) + alpha * e.getValue() * difference);
        }
    }

    private void feedback(State s, double val) {
        Action a; Counter newWeights = new Counter(weights);
        for (double multiplier = 1.0;
            multiplier > threshold && !history.empty();
            multiplier = multiplier * gamma) {
            a = history.pop();
            s.rewindTill(a);
            observeTransition(s, a, multiplier * (1.0 - gamma) * val, newWeights);
        }
        weights = newWeights;
    }

    private void positiveFeedback(State s) {
        feedback(s, 100.0);
    }

    private void negativeFeedback(State s) {
        feedback(s, -100.0);
    }

    private void neutralFeedback(State s) {
        feedback(s, 0.0);
    }

    public static void main(String[] args) {
        LearningAgent a = new LearningAgent(true);
        RandomAgent b = new RandomAgent(false);
        Agent[] agents = new Agent[]{a, b};
        Agent c;
        int numTrainingWins = 0, numTrainingLoses = 0,
            numTestingWins  = 0, numTestingLoses  = 0;

        System.out.println("Training begins.");
        for (int i = 0; i < numTraining; i++) {
            State s = new State();
            for (int j = 0; !s.end(); j = (j + 1) % 2) {
                c = agents[j];
                s.move(c.getAction(s));
            }
            if (s.blackWins()) {
                a.positiveFeedback(s);
                numTrainingWins++;
            } else if (s.whiteWins()) {
                a.negativeFeedback(s);
                numTrainingLoses++;
            } else {
                a.neutralFeedback(s);
            }
            if ((i + 1) % (numTraining / 10) == 0) {
                System.out.println((i + 1) / (numTraining / 10)
                    + "0% done. Wins/Loses: "
                    + numTrainingWins + "/" + numTrainingLoses);
                System.out.println(a.weights);
            }
        }
        a.doneTraining();
        System.out.println("Training completes. Testing begins.");
        for (int i = 0; i < numTesting; i++) {
            State s = new State();
            for (int j = 0; !s.end(); j = (j + 1) % 2) {
                c = agents[j];
                s.move(c.getAction(s));
            }
            if (s.blackWins()) {
                numTestingWins++;
            } else if (s.whiteWins()) {
                numTestingLoses++;
            }
        }
        System.out.println("Testing completes. Wins/Loses: "
            + numTestingWins + "/" + numTestingLoses);
        System.out.println(a.weights);
    }

}