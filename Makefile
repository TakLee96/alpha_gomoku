all:
	env mode=dev pypy main.py

debug:
	env mode=debug pypy main.py

test:
	env mode=test pypy test.py