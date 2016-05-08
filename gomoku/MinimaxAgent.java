package gomoku;

/** Advanced MinimaxAgent
 * @author TakLee96 */
public class MinimaxAgent extends ReflexAgent {

    private class Node {
        public Action a;
        public double v;
        public Node(Action a, double v) {
            this.a = a;
            this.v = v;
        }
    }

    public MinimaxAgent(boolean isBlack) {
        super(isBlack);
    }

    public double value(State s, int depth, int bigDepth) {
        // TODO
        return 0.0;
    }

    public Action[] getActions(State s) {
        // TODO
        return null;
    }

    @Override
    public Action getAction(State s) {
        // TODO
        return null;
    }

}
