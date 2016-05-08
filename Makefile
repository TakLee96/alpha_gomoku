all:
	java gomoku/GUIDriver

learn:
	java gomoku/LearningAgent

learn-out:
	java gomoku/LearningAgent > output.txt

visual:
	java gomoku/LearningAgent -v

compile:
	javac gomoku/*.java
