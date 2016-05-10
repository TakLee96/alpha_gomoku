package gomoku;

/** Game GUI Driver class
 * @author TakLee96 */
public class GUIDriver {

    private static final int N = State.N;

    private static int i2y(int i) { return i; }
    private static int y2i(int y) { return y; }
    private static int j2x(int j) { return N - j - 1; }
    private static int x2j(int x) { return N - x - 1; }

    private static String msg(State s) {
        if (s.ended())
            if (!s.isBlacksTurn()) return "Blue Naught wins!";
            else                   return "Green Cross wins!";
        else
            if (s.isBlacksTurn())  return "It's Blue Naught's turn!";
            else                   return "It's Green Cross's turn!";
    }

    public static void drawBoard(State s) {
        StdDrawPlus.clear();
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                StdDrawPlus.picture(i + .5, j + .5, s.get(j2x(j), i2y(i)).getImage(), 1, 1);
        StdDrawPlus.setPenColor(StdDrawPlus.DARK_GRAY);
        for (Action a : s.highlight)
            StdDrawPlus.square(y2i(a.y()) + .5, x2j(a.x()) + .5, .5);
        if (s.started()) {
            Action a = s.history.getLast();
            int newX = a.x(); int newY = a.y();
            if (s.inBound(newX, newY)) {
                StdDrawPlus.setPenColor(StdDrawPlus.RED);
                StdDrawPlus.square(y2i(newY) + .5, x2j(newX) + .5, .5);
            }
        }
        if (s.ended()) {
            StdDrawPlus.setPenColor(StdDrawPlus.ORANGE);
            for (Action t : s.five)
                StdDrawPlus.square(y2i(t.y()) + .5, x2j(t.x()) + .5, .5);
        }
        StdDrawPlus.setPenColor(StdDrawPlus.BLACK);
        StdDrawPlus.text(N / 2 + .5, N + .25, msg(s));
        StdDrawPlus.show(100);
    }

    public static void init() {
        StdDrawPlus.setXscale(0, N);
        StdDrawPlus.setYscale(0, N);
    }

    public static void main(String[] args) {
        init();
        State s = new State(); Action a = null;
        Agent agent = new MinimaxAgent(true);
        s.move(agent.getAction(s));

        while (true) {
            if (StdDrawPlus.mousePressed()) {
                double i = StdDrawPlus.mouseX();
                double j = StdDrawPlus.mouseY();
                int x = j2x((int) j), y = i2y((int) i);
                if (!s.ended() && s.canMove(x, y)) {
                    s.move(x, y);
                    drawBoard(s);
                    if (!s.ended()) {
                        System.out.println(s);
                        System.out.println("Feature: " + s.extractFeatures());
                        System.out.print("AI is thinking... ");
                        a = agent.getAction(s);
                        System.out.println("Done: " + a);
                        s.move(a);
                        System.out.println("=================================");
                    }
                }
            }
            drawBoard(s);
        }
    }
}
