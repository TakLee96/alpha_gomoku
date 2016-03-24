all:
	env mode=dev pypy main.py

debug:
	env mode=debug pypy main.py

test:
	env mode=test pypy test.py

test-detail:
	env mode=debug pypy test.py

test-once:
	env mode=test times=1 pypy test.py

test-once-detail:
	env mode=debug times=1 pypy test.py