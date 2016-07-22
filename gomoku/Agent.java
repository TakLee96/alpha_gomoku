package gomoku;

/** Agent interface playing randomly
 * @author TakLee96 */
public class Agent {

    public boolean isBlack;
    public Agent(boolean isBlack) {
        this.isBlack = isBlack;
    }

    public Action getAction(State s) {
        if (!s.started()) return s.start;
        Action a = s.randomAction();
        System.out.println("Done: " + a);
        return a;
    }

}
