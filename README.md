# Alpha Gomoku

- This is a gomoku game with Java GUI
- An AI is developed with Minimax Search
- You will always play as White
- Double Threat is allowed for Black
- Six stones do not win the game

## Wiki

Please have a look at the [wiki](https://github.com/TakLee96/alpha_gomoku/wiki) page
to learn about [how does the AI work](https://github.com/TakLee96/alpha_gomoku/wiki#how-does-the-ai-work)
and about [Java API](https://github.com/TakLee96/alpha_gomoku/wiki/Java-API) you can use.

## To install and play

Have JRE 8 installed and then (choose your way):
1. Run in terminal or cmd `java -jar gomoku.jar`
2. Double click `gomoku.jar`
3. Or start a static server and navigate to **http://localhost:8000**
  + `python -m SimpleHTTPServer 8000`
  + `python3 -m http.server 8000`
4. Have JDK 8 installed and run in terminal `make build run`

## Old version

- Have **pypy** and **bottle** installed and run `pypy old/main.py`
- Navigate your browser to `http://localhost:8000`
- The old-version AI might be a bit slow, but it's not too bad
- The website can also read a history JSON object and display an old game, like `[(7, 7), (9, 6), ...]`

## Potential Improvements

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
  + Could keep HashMap in UI thread
