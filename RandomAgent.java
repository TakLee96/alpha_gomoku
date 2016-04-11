public class RandomAgent extends Agent {

    public RandomAgent(boolean isBlack) {
        super(isBlack);
    }

    public Action getAction(State s) {
        return s.randomAction();
    }

}