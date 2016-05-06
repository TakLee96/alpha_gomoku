all:
	java gomoku/GUIDriver

learn:
	java gomoku/LearningAgent

visual:
	java gomoku/LearningAgent -v

compile:
	javac gomoku/*.java
