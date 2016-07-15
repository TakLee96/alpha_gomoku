all:
	@echo "[make] build/jar/run"

run:
	@java gomoku/App

build:
	@javac gomoku/*.java
	@echo "[make] class files recompiled"

jar:
	@jar cfe gomoku.jar gomoku.App gomoku
	@echo "[make] gomoku.jar created"

