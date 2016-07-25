package gomoku;

import java.util.Set;
import java.util.List;
import java.util.HashSet;
import java.util.HashMap;
import java.util.ArrayList;

import javax.swing.SwingUtilities;

/** Parallel MinimaxAgent
 * @author TakLee96 */
public class ParallelAgent extends MinimaxAgent {

    private static final int NUM_THREADS = 3;

    private static class AsyncMap<K, V> extends HashMap<K, V> {
        @Override public V put(K key, V val) {
            SwingUtilities.invokeLater(new Runnable() {
                @Override
                public void run() {
                    AsyncMap.super.put(key, val);
                }
            });
            return null;
        }  
    }

    private class DelegateAgent implements Runnable {
        private State state;
        private Set<Action> actions;
        private Node node;
        public DelegateAgent(State s, List<Action> a) {
            state = s.clone();
            actions = new HashSet<Action>(a);
            node = null;
        }
        @Override
        public void run() {
            boolean who = state.isBlacksTurn();
            if (who) node = maxvalue(state, -infinity, infinity, 1, 0, actions);
            else     node = minvalue(state, -infinity, infinity, 1, 0, actions);
        }
    }

    public ParallelAgent(boolean isBlack) {
        super(isBlack);
        memo = new AsyncMap<Set<Move>, Node>();
    }

    @Override
    public Action getAction(State s) {
        if (!s.isTurn(isBlack))
            throw new RuntimeException("not my turn");
        Node retval = null;
        if (!s.started())
            return s.start;
        memo.clear();
        Set<Action> a = getActions(s);
        if (a.size() < bigDepthThreshold) {
            retval = value(s, -infinity, infinity, 0, 0);
        } else {
            s.highlight(a);
            List<Action> actions = new ArrayList<Action>(a);
            int workPerThread = actions.size() / NUM_THREADS;
            Thread[] threads = new Thread[NUM_THREADS];
            DelegateAgent[] agents = new DelegateAgent[NUM_THREADS];
            for (int i = 0; i < NUM_THREADS; i++) {
                agents[i] = new DelegateAgent(s, actions.subList(i * workPerThread, (i + 1) * workPerThread));
            }
            for (int j = NUM_THREADS * workPerThread; j < a.size(); j++) {
                agents[j % NUM_THREADS].actions.add(actions.get(j));
            }
            for (int i = 0; i < NUM_THREADS; i++) {
                threads[i] = new Thread(agents[i]);
                threads[i].start();
            }
            boolean allReady = false;
            while (!allReady) {
                allReady = true;
                for (int i = 0; i < NUM_THREADS; i++) {
                    allReady = allReady && !threads[i].isAlive();
                }
                try { Thread.sleep(100); } catch (Exception e) {}
            }
            if (isBlack) {
                retval = agents[0].node;
                for (int i = 1; i < NUM_THREADS; i++)
                    if (agents[i].node.v > retval.v) 
                        retval = agents[i].node;
            } else {
                retval = agents[0].node;
                for (int i = 1; i < NUM_THREADS; i++)
                    if (agents[i].node.v < retval.v) 
                        retval = agents[i].node;
            }
        }
        s.unhighlight();
        return retval.a;
    }

}
