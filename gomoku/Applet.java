package gomoku;

import javax.swing.JApplet;
import javax.swing.JButton;
import javax.swing.ImageIcon;
import javax.swing.BorderFactory;
import javax.swing.border.Border;
import javax.swing.SwingUtilities;

import java.awt.Container;
import java.awt.Component;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.event.ActionEvent;

import java.net.URL;
import java.util.Set;

/** HTML Embedded Java Applet
 * @author TakLee96 */
public class Applet extends JApplet {
    private static final int L = 50;
    private static final int N = State.N;
    private static final int S = L * N;
    private static Border redBorder = BorderFactory.createLineBorder(Color.red);
    private static Border yellowBorder = BorderFactory.createLineBorder(Color.yellow);
    private boolean isBlacksTurn = true;
    private boolean uiready = false;
    private ImageIcon blue   = null,
                      empty  = null,
                      green  = null,
                      red    = null,
                      yellow = null;
    private JButton[][] ref;
    private State state = new State();;
    private MinimaxAgent agent = new MinimaxAgent(true);
    private static void sleep(long time) {
        try {
            Thread.sleep(time);
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }
    }
    private void click(Action a) {
        click(a.x(), a.y());
    }
    private void click(int i, int j) {
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
    private class ButtonListener implements java.awt.event.ActionListener {
        private int i, j;
        public ButtonListener(int x, int y) { i = x; j = y; }
        @Override
        public void actionPerformed(ActionEvent e) {
            if (!state.ended() && state.canMove(i, j) && !isBlacksTurn) {
                click(i, j);
            }
        }
    }

    @Override
    public void init() {
        String base = this.getDocumentBase().toString();
        if (base.indexOf('#') != -1) base = base.substring(0, base.indexOf('#'));
        if (base.indexOf(".html") != -1) base = base.substring(0, base.lastIndexOf('/'));
        if (base.charAt(base.length()-1) != '/') base += "/";
        base += "gomoku/img/";
        try {
            blue   = new ImageIcon(this.getImage(new URL(base+"grid-blue.jpg")));
            empty  = new ImageIcon(this.getImage(new URL(base+"grid-empty.jpg")));
            green  = new ImageIcon(this.getImage(new URL(base+"grid-green.jpg")));
            red    = new ImageIcon(this.getImage(new URL(base+"grid-red.jpg")));
            yellow = new ImageIcon(this.getImage(new URL(base+"grid-yellow.jpg")));
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }

        ref = new JButton[N][N];
        Container content = super.getContentPane();
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
        uiready = true;
    }

    private class AIRunner implements Runnable {
        @Override
        public void run() {
            state.onHighlight(new ActionsListener() {
                @Override public void digest(Set<Action> actions) {
                    int i = 0, j = 0;
                    for (Action a : actions) {
                        i = a.x();
                        j = a.y();
                        ref[i][j].setIcon(yellow);
                    }
                }
            });
            state.onUnhighlight(new ActionsListener() {
                @Override public void digest(Set<Action> actions) {
                    int i = 0, j = 0;
                    for (Action a : actions) {
                        i = a.x();
                        j = a.y();
                        ref[i][j].setIcon(empty);
                    }
                }
            });
            state.onEvaluate(new ActionListener() {
                @Override public void digest(Action action) {
                    ref[action.x()][action.y()].setIcon(red);
                }
            });
            state.onDetermineMove(new ActionListener() {
                @Override public void digest(Action action) {
                    click(action);
                }
            });
            while (!uiready) {
                sleep(100);
            }
            while (!state.ended()) {
                if (isBlacksTurn) {
                    agent.getAction(state);
                }
                sleep(100);
            }
        }
    }

    @Override
    public void start() {
        Thread gomoku = new Thread(new AIRunner());
        gomoku.start();
    }
}
