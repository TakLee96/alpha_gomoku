package gomoku;

/** Agent playing randomly for the game
 * @author TakLee96 */
public class Agent {

    public boolean isBlack;
    public Agent(boolean isBlack) {
        this.isBlack = isBlack;
    }

    public Action getAction(State s) {
        if (!s.started()) return s.start;
        return s.randomAction();
    }

}
