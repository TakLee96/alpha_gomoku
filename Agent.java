public abstract class Agent {

    protected boolean isBlack;
    public Agent(boolean isBlack) {
        this.isBlack = isBlack;
    }

    public abstract Action getAction(State s);

}