package gomoku;

import java.util.ArrayDeque;

/** Helper class for State
 * @author TakLee96 */
class Rewinder {

    public ArrayDeque<Action> removedLegalActions;
    public Counter diffFeatures;
    public Rewinder(ArrayDeque<Action> rla, Counter df) {
        removedLegalActions = rla;
        diffFeatures = df;
    }

}
