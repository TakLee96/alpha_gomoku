# Alpha Gomoku

- This is a standard gomoku game with Java GUI (the AI is currently playing randomly)

# To install and play

- Have Java installed and run `javac *.java && java Driver`

# Old version

- Have **pypy** and **bottle** installed and run `pypy old/main.py`
- Navigate your browser to `localhost:8000`
- The AI is working, and fairly strong I believe
- The website can read a history JSON object and replay an old game

# Past problems

- Once winning strategy is found, can cache responses
- Not intelligent enough during open status
- Still have redundant calculation
- Big Calculation is not perfect
- Diff feature calculating is probably better than extract feature for each state
- Could apply adversarial reinforcement learning
- Could apply threat space search
