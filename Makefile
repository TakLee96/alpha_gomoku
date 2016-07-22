all:
	@echo "[make] Usage: make [jar/run/build/clean]"
	@echo "jar - generate gomoku.jar file"
	@echo "run - launch gomoku application"
	@echo "build - compile java files to class files"
	@echo "clean - remove jar and class files"

benchmark:
	@echo "[make] running benchmark"
	@java gomoku/Benchmark

run:
	@echo "[make] launching application"
	@java gomoku/App

build:
	@javac gomoku/*.java
	@echo "[make] class files recompiled"

jar:
	@jar cfe gomoku.jar gomoku.App gomoku
	@echo "[make] gomoku.jar created"

clean-class:
	@rm gomoku/*.class
	@echo "[make] class files cleaned"

clean-jar:
	@rm gomoku.jar
	@echo "[make] jar file cleaned"

clean: clean-class clean-jar

rebuild: clean-class build
