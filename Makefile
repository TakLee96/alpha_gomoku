all:
	env host=0.0.0.0 port=80 nohup pypy main.py > output.txt
