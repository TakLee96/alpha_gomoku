public class Driver {

    private static final int N = State.N;

    private static int i2y(int i) { return i; }
    private static int y2i(int y) { return y; }
    private static int j2x(int j) { return N - j - 1; }
    private static int x2j(int x) { return N - x - 1; }

    public static void drawBoard(State s) {
        StdDrawPlus.clear();
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                StdDrawPlus.picture(i + .5, j + .5, s.get(j2x(j), i2y(i)).getImage(), 1, 1);
            }
        }
        if (s.inBound(s.newX, s.newY)) {
            StdDrawPlus.setPenColor(StdDrawPlus.RED);
            StdDrawPlus.square(y2i(s.newY) + .5, x2j(s.newX) + .5, .5);
        }
        if (s.end()) {
            StdDrawPlus.setPenColor(StdDrawPlus.ORANGE);
            for (Action t : s.five) {
                StdDrawPlus.square(y2i(t.y()) + .5, x2j(t.x()) + .5, .5);
            }
        }
        StdDrawPlus.setPenColor(StdDrawPlus.BLACK);
        StdDrawPlus.text(N / 2 + .5, N + .25, s.message);
    }

    public static void main(String[] args) {
        StdDrawPlus.setXscale(0, N);
        StdDrawPlus.setYscale(0, N);
        State s = new State();
        Agent agent = new RandomAgent(true);
        s.move(agent.getAction(s));

        while (true) {
            if (StdDrawPlus.mousePressed()) {
                double i = StdDrawPlus.mouseX();
                double j = StdDrawPlus.mouseY();
                int x = j2x((int) j), y = i2y((int) i);
                if (s.canMove(x, y)) {
                    s.move(x, y);
                    Action a = agent.getAction(s);
                    if (s.canMove(a)) {
                        s.move(a);
                        System.out.println(s);
                        System.out.println(s.extractFeatures());
                    }
                }
            }
            drawBoard(s);
            StdDrawPlus.show(100);
        }
    }
}
