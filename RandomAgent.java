public class RandomAgent extends Agent {

    public RandomAgent(boolean isBlack) {
        super(isBlack);
    }

    public Action getAction(State s) {
        if (!s.started()) return s.start;
        return s.randomAction();
    }

}
