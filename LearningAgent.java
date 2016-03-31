public class LearningAgent {

    double alpha, gamma, epsilon;
    Counter weights;
    public LearningAgent(double a, double g, double e) {
        alpha = a;
        gamma = g;
        epsilon = e;
        weights = new Counter();
    }

    public static void main(String[] args) {
        LearningAgent a = new LearningAgent();

        for (int i = 0; i < numTraining; i++) {
            State s = new State();
        }
    }

}