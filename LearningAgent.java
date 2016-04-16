import java.util.Random;
import java.util.Map;

public class LearningAgent extends Agent {

    private static final double alpha = 0.1;
    private static final double gamma = 0.8;
    private static final double epsilon = 0.1;
    private static final double reward = 100.0;
    private static final int numTraining = 100;
    private static final int numTesting = 10;

    private Counter weights;
    private boolean doneTraining;
    private Action prev;
    private Random random;
    public LearningAgent(boolean isBlack) {
        super(isBlack);
        weights = new Counter();
        doneTraining = false;
        prev = null;
        random = new Random(System.currentTimeMillis());
    }

    private double computeValue(State s) {
        Map<String, Integer> features = s.extractFeatures();
        double val = 0.0;
        for (Map.Entry<String, Integer> e : features.entrySet())
            val += e.getValue() * weights.get(e.getKey());
        return val;
    }

    private double computeQValue(State s, Action a) {
        double val = 0.0;
        double minval = Double.MAX_VALUE;
        s.move(a);
        if (s.win(isBlack)) {
            s.rewind();
            return reward;
        }
        for (Action ap : s.getLegalActions()) {
            s.move(ap);
            if (s.win(!isBlack)) {
                s.rewind(); s.rewind();
                return -reward;
            }
            val = computeValue(s);
            if (val < minval) minval = val;
            s.rewind();
        }
        s.rewind();
        return minval;
    }

    private Action getPolicy(State s) {
        double val = 0.0;
        double maxval = -Double.MAX_VALUE;
        Action maxaction = null;
        for (Action a : s.getLegalActions()) {
            val = computeQValue(s, a);
            if (val > maxval) {
                maxval = val;
                maxaction = a;
            }
        }
        return maxaction;
    }

    @Override
    public Action getAction(State s) {
        if (!s.started()) {
            prev = s.start;
        } else if (doneTraining || random.nextDouble() > epsilon) {
            prev = getPolicy(s);
        } else {
            prev = s.randomAction();
        }
        return prev;
    }

    private void doneTraining() {
        doneTraining = true;
    }

    private void observeTransition(State s, Action a, double r) {
        Map<String, Integer> features = s.extractFeatures();
        double difference;
        for (Map.Entry<String, Integer> e : features.entrySet()) {
            difference = r - computeValue(s) + gamma * computeQValue(s, a);
            weights.put(e.getKey(), weights.get(e.getKey()) + alpha * difference * e.getValue());
        }
    }

    public static void main(String[] args) {
        LearningAgent first  = new LearningAgent(true);
        LearningAgent second = new LearningAgent(false);
        LearningAgent[] agents = new LearningAgent[]{first, second};
        LearningAgent agent;
        int numTrainingWins = 0, numTrainingLoses = 0,
            numTestingWins  = 0, numTestingLoses  = 0;
        int nx, ny, px = 0, py = 0; boolean started;

        System.out.println("Training begins.");
        for (int i = 0; i < numTraining; i++) {
            State s = new State();
            first.prev = null; second.prev = null;
            for (int j = 0; !s.end(); j = (j + 1) % 2) {
                agent = agents[j];
                if (agent.prev != null) {
                    nx = s.newX; ny = s.newY; s.rewind(); started = s.started();
                    if (started) { px = s.newX; py = s.newY; s.rewind(); }
                    agent.observeTransition(s, agent.prev, 0.0);
                    if (started) s.move(px, py);
                    s.move(nx, ny);
                }
                s.move(agent.getAction(s));
                System.out.print(".");
            }
            if (s.blackWins()) {
                first.observeTransition(s, first.prev, reward);
                s.rewind();
                second.observeTransition(s, second.prev, -reward);
                numTrainingWins++;
                System.out.println("o");
            } else if (s.whiteWins()) {
                second.observeTransition(s, second.prev, reward);
                s.rewind();
                first.observeTransition(s, first.prev, -reward);
                numTrainingLoses++;
                System.out.println("x");
            } else {
                System.out.println(first.weights);
                System.out.println(second.weights);
                throw new RuntimeException("Even?!?!");
            }
            if ((i + 1) % (numTraining / 10) == 0) {
                System.out.println((i + 1) / (numTraining / 10)
                    + "0% done. First's Wins/Loses: "
                    + numTrainingWins + "/" + numTrainingLoses);
                System.out.println(first.weights);
                System.out.println(second.weights);
            }
        }
        first.doneTraining(); second.doneTraining();
        System.out.println("Training completes. Testing begins.");
        for (int i = 0; i < numTesting; i++) {
            State s = new State();
            for (int j = 0; !s.end(); j = (j + 1) % 2) {
                agent = agents[j];
                s.move(agent.getAction(s));
            }
            if (s.blackWins()) {
                numTestingWins++;
            } else if (s.whiteWins()) {
                numTestingLoses++;
            }
        }
        System.out.println("Testing completes. First's Wins/Loses: "
            + numTestingWins + "/" + numTestingLoses);
        System.out.println(first.weights);
        System.out.println(second.weights);
    }

}
