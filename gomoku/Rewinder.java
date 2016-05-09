package gomoku;

import java.util.LinkedList;

/** Helper class for State
 * @author TakLee96 */
public class Rewinder {

    public LinkedList<Action> removedLegalActions;
    public Counter diffFeatures;
    public Rewinder(LinkedList<Action> rla, Counter df) {
        removedLegalActions = rla;
        diffFeatures = df;
    }

}
