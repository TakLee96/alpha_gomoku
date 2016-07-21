package gomoku;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JApplet;
import javax.swing.JButton;
import javax.swing.ImageIcon;
import javax.swing.BorderFactory;
import javax.swing.border.Border;
import javax.swing.SwingUtilities;

import java.awt.Color;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import java.io.File;
import java.util.Set;

public class App {
    private static final int L = 50;
    private static final int N = State.N;
    private static final int S = L * N;
    private static Border redBorder = BorderFactory.createLineBorder(Color.red);
    private static Border yellowBorder = BorderFactory.createLineBorder(Color.yellow);
    private static JFrame frame = new JFrame("Alpha Gomoku");
    private static ImageIcon blue   = null,
                             empty  = null,
                             green  = null,
                             red    = null,
                             yellow = null;
    private static boolean uiready = false;
    private static boolean isBlacksTurn = true;
    private static JButton[][] ref;
    private static State state = new State();
    private static MinimaxAgent agent = new MinimaxAgent(true);
    private static void sleep(long time) {
        try {
            Thread.sleep(time);
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }
    }
    private static void click(Action a) {
        click(a.x(), a.y());
    }
    private static void click(int i, int j) {
        if (state.started()) {
            Action a = state.lastAction();
            ref[a.x()][a.y()].setBorderPainted(false);
        }
        state.move(i, j);
        isBlacksTurn = !isBlacksTurn;
        if (state.get(i, j).isBlack())
            ref[i][j].setIcon(blue);
        else ref[i][j].setIcon(green);
        ref[i][j].setBorder(redBorder);
        ref[i][j].setBorderPainted(true);
        if (state.ended()) {
            for (Action a : state.five) {
                ref[a.x()][a.y()].setBorder(yellowBorder);
                ref[a.x()][a.y()].setBorderPainted(true);
            }
        }
    }
    private static class ButtonListener implements ActionListener {
        private int i, j;
        public ButtonListener(int x, int y) { i = x; j = y; }
        @Override
        public void actionPerformed(ActionEvent e) {
            if (!state.ended() && state.canMove(i, j) && !isBlacksTurn) {
                click(i, j);
            }
        }
    }

    public static void init(Container content) {
        String base = "gomoku/img/";
        try {
            blue   = new ImageIcon(ImageIO.read(new File(base + "grid-blue.jpg")));
            empty  = new ImageIcon(ImageIO.read(new File(base + "grid-empty.jpg")));
            green  = new ImageIcon(ImageIO.read(new File(base + "grid-green.jpg")));
            red    = new ImageIcon(ImageIO.read(new File(base + "grid-red.jpg")));
            yellow = new ImageIcon(ImageIO.read(new File(base + "grid-yellow.jpg")));
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }

        ref = new JButton[N][N];
        content.setBackground(Color.white);

        JButton grid = null;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                grid = new JButton(empty);
                grid.setLocation(j * L, i * L);
                grid.setSize(L, L);
                grid.addActionListener(new ButtonListener(i, j));
                ref[i][j] = grid;
                content.add(grid);
            }
        }

        // trigger UI update
        grid = new JButton();
        content.add(grid);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.setResizable(false);
                Container main = frame.getContentPane();
                Dimension dim = new Dimension(S, S);
                main.setPreferredSize(dim);
                init(frame.getContentPane());
                frame.pack();
                frame.setVisible(true);
                uiready = true;
            }
        });

        state.onHighlight(new Listener() {
            @Override
            public void digest(Set<Action> actions) {
                int i = 0, j = 0;
                for (Action a : actions) {
                    i = a.x();
                    j = a.y();
                    ref[i][j].setIcon(yellow);
                }
            }
        });
        state.onUnhighlight(new Listener() {
            @Override
            public void digest(Set<Action> actions) {
                int i = 0, j = 0;
                for (Action a : actions) {
                    i = a.x();
                    j = a.y();
                    ref[i][j].setIcon(empty);
                }
            }
        });
        state.onEvaluate(new Listener() {
            @Override
            public void digest(Set<Action> actions) {
                int i = 0, j = 0;
                for (Action a : actions) {
                    i = a.x();
                    j = a.y();
                    ref[i][j].setIcon(red);
                }
            }
        });

        while (!uiready) {
            sleep(100);
        }
        while (!state.ended()) {
            if (isBlacksTurn) {
                click(agent.getAction(state));
            }
            sleep(100);
        }
    }
}
