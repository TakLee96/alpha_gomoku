package gomoku;

import java.text.DecimalFormat;

/** Advanced MinimaxAgent
 * @author TakLee96 */
public class Benchmark {
    private static DecimalFormat f = new DecimalFormat("#0.000");
    public static void main(String[] args) {
        State state = new State();
        MinimaxAgent agent = new MinimaxAgent(false);
        int[] moves = new int[]{
            7, 7, 7, 6, 8, 6, 8, 5, 8, 7, 9, 5, 9, 6,
            7, 5,10, 5, 7, 8, 8, 8, 8, 9, 6, 7, 9, 7,
            6, 6, 9, 9,10, 6,11, 6,10, 7,10, 4, 6, 8
        };
        for (int i = 0; i < moves.length; i += 2) {
            state.move(moves[i], moves[i+1]);
        }
        System.out.println(state);
        System.out.println("begin thinking");
        long time = System.currentTimeMillis();
        agent.getAction(state);
        System.out.println("done thinking");
        System.out.println("time ellapsed: " + (System.currentTimeMillis() - time) + "ms");
        System.out.println("====== Detailed Analysis ======");
        System.out.println("=> total eval: " + (agent.numInstantEval + agent.numDepthEval));
        System.out.println("====> instant eval: " + agent.numInstantEval);
        System.out.println("====> depth eval: " + agent.numDepthEval);
        System.out.println("====> average depth: " + f.format(1.0 * agent.totalEvalDepth / agent.numDepthEval));
        System.out.println("=> cache performance: " + f.format(100.0 * agent.numCacheHit / (agent.numCacheHit + agent.numCacheMiss)) + "%");
        System.out.println("====> cache hit: " + agent.numCacheHit);
        System.out.println("====> cache miss: " + agent.numCacheMiss);
        System.out.println("=> average branch: " + f.format(1.0 * agent.totalBranchSize / agent.numRecursion));
    }
}