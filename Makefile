all:
	@echo "Usage: make [compile/black/white/test]"
	@echo "- compile: compile all the java files (do this first)"
	@echo "- black:   start the gomoku game as black"
	@echo "- white:   start the gomoku game as white"
	@echo "- test:    watch a game played by two AI agents"

compile:
	@javac gomoku/*.java
	@echo "Done."

black:
	@java gomoku/GUIDriver black

white:
	@java gomoku/GUIDriver
	
test:
	@java gomoku/MinimaxAgent

