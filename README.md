# Alpha Gomoku

- This is a standard gomoku game with Java GUI

# To install and play

- Have Java 8 installed
- Run `javac gomoku/*.java`
- Start a static server
  + `python -m SimpleHTTPServer 8000`
  + `python3 -m http.server 8000`
- Navigate to **http://localhost:8000**

# Java API

If you want to make your own gomoku game, or make use of this AI,
you can include this package `gomoku`.

## Action.java

```java
public Action(int x, int y);
public int x();
public int y();
```

## Agent.java

**MinimaxAgent.java** extends this class.

```java
public Agent(boolean isBlack);
public Action getAction(State s);
```

## Listener.java

Interface useful for updating UI.

```java
public void digest(Set<Action> actions);
```

## State.java

```java
// Core Utility
public int numMoves();
public boolean started();
public boolean ended();
public boolean win(boolean isBlack);
public void move(Action a);
// UI Life Cycle
public void onHighlight(Listener listener); // considering these locations
public void onEvaluate(Listener listener); // evaluated this location
public void onUnhighlight(Listener listener); // done thinking
```

# Old version

- Have **pypy** and **bottle** installed and run `pypy old/main.py`
- Navigate your browser to `http://localhost:8000`
- The old-version AI might be a bit slow, but it's not too bad
- The website can also read a history JSON object and display an old game, like `[(7, 7), (9, 6), ...]`

# Past problems

- Once winning strategy is found, can cache responses with a Trie
- Can still improve action picking heuristic
- Not intelligent enough during open status
- Still might have redundant calculation
- Could apply adversarial reinforcement learning
