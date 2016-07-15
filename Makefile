all:
	@echo "[Make] Usage: make [jar/run/build/clean]"
	@echo "jar - generate gomoku.jar file"
	@echo "run - launch gomoku application"
	@echo "build - compile java files to class files"
	@echo "clean - remove jar and class files"

run:
	@echo "[Make] launching application"
	@java gomoku/App

build:
	@javac gomoku/*.java
	@echo "[make] class files recompiled"

jar:
	@jar cfe gomoku.jar gomoku.App gomoku
	@echo "[make] gomoku.jar created"

clean:
	@rm gomoku/*.class
	@rm gomoku.jar
	@echo "[make] class files and jar file removed"
