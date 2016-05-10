# Alpha Gomoku

- This is a standard gomoku game with Java GUI

# To install and play

- Have Java 7 installed, run `make compile`, and then `make`

# Old version

- Have **pypy** and **bottle** installed and run `pypy old/main.py`
- Navigate your browser to `localhost:8000`
- The old-version AI might be a bit slow, but it's strong
- The website can also read a history JSON object and display an old game, like `[(7, 7), (9, 6), ...]`

# Past problems

- Once winning strategy is found, can cache responses with a Trie
- Can still improve action picking heuristic
- Not intelligent enough during open status
- Still might have redundant calculation
- Could apply adversarial reinforcement learning
