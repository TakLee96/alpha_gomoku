package gomoku;

/** Advanced MinimaxAgent
 * @author TakLee96 */
public class MinimaxAgent extends Agent {

    public MinimaxAgent(boolean isBlack) {
        super(isBlack);
    }

    @Override
    public Action getAction(State s) {
        if (!s.started()) return s.start;
        return s.randomAction();
    }

}
