all:
	@echo "Usage: make [compile/black/white]"
	@echo "- compile: compile all the java files (do this first)"
	@echo "- black: start the gomoku game as black"
	@echo "- white: start the gomoku game as white"

compile:
	@javac gomoku/*.java
	@echo "Done."

black:
	@java gomoku/GUIDriver black

white:
	@java gomoku/GUIDriver

