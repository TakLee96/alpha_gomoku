all:
	env host=0.0.0.0 port=80 nohup pypy main.py > output.txt

install:
	killall pypy
	git pull origin master
	env host=0.0.0.0 port=80 nohup pypy main.py > output.txt

