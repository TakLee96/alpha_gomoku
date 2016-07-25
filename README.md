# Alpha Gomoku

- This is a standard gomoku game with Java GUI

# To install and play

- Have JDK SE 8 installed
- Compile the source code by `javac gomoku/*.java`
- Start a static server and navigate to **http://localhost:8000**
  + `python -m SimpleHTTPServer 8000`
  + `python3 -m http.server 8000`
- Or simply run `java gomoku/App`

# Java API

If you want to design your own gomoku game, make use of this AI, or
write your own AI that is compatible with this package, please read through
the brief API description about package `gomoku`.

## Action.java

Simple (x, y) tuple.

```java
public Action(int x, int y);
public int x();
public int y();
```

## Agent.java

The main AI **MinimaxAgent.java** extends this class.

```java
public Agent(boolean isBlack);
public Action getAction(State s);
```

## App.java

The Swing Application. Entrance of this package.

## Applet.java

The HTML embeddable Applet used in `index.html`.

## Counter.java

A `Map<String, Number>` with default value 0 (delegate to HashMap).

## Extractor.java

For internal use, to extract feature for Gomoku game.

## Grid.java

The Gomoku board is made up of a matrix of Grid object.

```java
public boolean isBlack();
public boolean isWhite();
public boolean is(boolean isBlack);
public boolean isEmpty();
public void put(boolean isBlack);
public void clean();
```

## Listener.java

Set callback hooks on `State` object with `Listener`, and `MinimaxAgent` will
invoke these callbacks. An interface useful for updating UI. 

```java
public void digest(Set<Action> actions);
```

## MinimaxAgent.java

Core AI functionality which extends `Agent`. See `Agent.java` for usage.

## Move.java

For internal use, to keep track of x and y, as well as who.

## Rewinder.java

For internal use, to keep enough information to rewind moves in `State`.

## State.java

Keep track of the Gomoku game state with useful utility functions.

```java
// Core Utility
public int numMoves();
public boolean started();
public boolean isBlacksTurn();
public boolean isTurn(boolean isBlack);
public boolean canMove(Action a);
public boolean canMove(int x, int y);
public void move(Action a);
public void move(int x, int y);
public Grid get(Action a);
public Grid get(int x, int y);
public boolean inBound(Action a);
public boolean inBound(int x, int y);
public boolean win(boolean isBlack);
public boolean ended();
public Action[] getLegalActions();
public Action randomAction();
public Action lastAction();
public Counter extractFeatures();
public ArrayDeque<Action> history();
public String toString();

// AI Life Cycle (For writing your AI)
public void highlight(Set<Action> actions);
public void evaluate(Action a);
public void unhighlight();

// UI Life Cycle (For writing your App)
public void onHighlight(Listener listener); // considering these locations
public void onEvaluate(Listener listener); // evaluated this location
public void onUnhighlight(Listener listener); // done thinking
```

# Old version

- Have **pypy** and **bottle** installed and run `pypy old/main.py`
- Navigate your browser to `http://localhost:8000`
- The old-version AI might be a bit slow, but it's not too bad
- The website can also read a history JSON object and display an old game, like `[(7, 7), (9, 6), ...]`

# Potential Improvements

- Cache winning strategy with Trie
- Fine tune parameters
  + Design action picking weights
  + Tune evaluation weights
  + Tune hyperparams: `maxDepth`, `maxBigDepth`, `bigDepthThreshold`, `branch`
- Reduce redundant calculation
- Style consistency computation (just aggressive vs just defensive)
- Parallel computing
  + ConcurrentHashMap might be blocking performance
  + Could let each thread use its own cache
