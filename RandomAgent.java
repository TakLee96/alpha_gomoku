import java.util.Set;
import java.util.Random;

public class RandomAgent extends Agent {

    private Random random;
    public RandomAgent(boolean isBlack) {
        super(isBlack);
        random = new Random();
    }

    public Action getAction(State s) {
        Set<Action> set = s.getLegalActions();
        int i = 0, index = random.nextInt(set.size());
        for (Action a : set) {
            if (i == index) {
                return a;
            }
            i++;
        }
        return null;
    }

}