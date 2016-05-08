package gomoku;

import java.util.ArrayList;
import java.util.Random;
import java.util.Map;
import java.io.BufferedWriter;
import java.io.FileWriter;

/** Reinforcement Learning Agent
 * @author TakLee96 */
public class LearningAgent extends ReflexAgent {

    private static final int N = State.N;
    private static final double alpha = 0.2;
    private static final double epsilon = 0.1;
    private static final double reward = 1.0;
    private static final double maxStepSize = 100.0;
    private static final int numTraining = 20;
    private static final int numTesting = 0;
    private static final Random random = new Random();

    private static double alpha(int episodes) {
        return Math.min(alpha / Math.log(episodes + 0.1), alpha);
    }

    private boolean doneTraining;
    public LearningAgent(boolean isBlack) {
        super(isBlack, new Counter());
        doneTraining = false;
    }

    // TODO: be careful with this
    private void observe(State s, double reward, int episodes) {
        Action a = s.rewind(null); // error!!!
        Map<String, Integer> features = s.extractFeatures();
        double delta = reward - value(s, features) + gamma * nextValue(s);
        delta = Math.signum(delta) * Math.min(Math.abs(delta), maxStepSize);
        for (String key : features.keySet())
            weights.put(key, weights.get(key) + alpha(episodes) * delta * features.get(key));
        s.move(a);
        if (!isBlack)
            System.out.println(a + " " + delta);
    }

    private void doneTraining() {
        doneTraining = true;
    }

    @Override
    public Action getAction(State s) {
        if (doneTraining || !s.started() || random.nextDouble() > epsilon)
            return getPolicy(s);
        return s.randomAction();
    }

    private static void playGame(LearningAgent b, LearningAgent w, boolean withGraphics, int episodes) {
        State s = new State();
        System.out.print(episodes);
        while (true) {
            if (s.isBlacksTurn()) {
                s.move(b.getAction(s));
            } else {
                s.move(w.getAction(s));
            }
            if (withGraphics) GUIDriver.drawBoard(s);
            System.out.print(".");
            if (!s.ended()) {
                b.observe(s, 0.0, episodes);
                w.observe(s, 0.0, episodes);
            } else break;
        }
        double blackReward = 0.0;
        if (s.win(true)) {
            System.out.println("win");
            blackReward = reward;
        } else if (s.win(false)) {
            System.out.println("lose");
            blackReward = -reward;
        } else {
            System.out.println("tie");
            blackReward = 0.0;
        }
        b.observe(s, blackReward, episodes);
        w.observe(s, blackReward, episodes);
        System.out.println(w.weights);
        if (withGraphics) System.out.println(s.history);
    }

    public static void main(String[] args) {
        boolean withGraphics = args != null && args.length > 0;
        LearningAgent b = new LearningAgent(true);
        LearningAgent w = new LearningAgent(false);
        if (withGraphics) GUIDriver.init();

        for (int i = 1; i <= numTraining; i++) {
            playGame(b, w, withGraphics, i);
        }

        System.out.println(b.weights);
        b.doneTraining();
        w.doneTraining();
        if (!withGraphics && numTesting > 0) GUIDriver.init();
        for (int j = 0; j < numTesting; j++) {
            playGame(b, w, true, 0);
        }

        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter("gomoku/weight.csv"));
            String weight = b.weights.toString();
            bw.write(weight, 0, weight.length());
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.exit(0);
    }

}
