public abstract class Agent {

    public boolean isBlack;
    public Agent(boolean isBlack) {
        this.isBlack = isBlack;
    }

    public abstract Action getAction(State s);

}
