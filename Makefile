all:
	pypy main.py

debug:
	env mode=debug pypy main.py

test:
	pypy test.py