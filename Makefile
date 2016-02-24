all:
	env host=localhost port=8000 pypy main.py > output.txt

server:
	env host=0.0.0.0 port=80 nohup pypy main.py > output.txt

update:
	killall pypy
	git pull origin master

install:
	brew install pypy
	pypy -m ensurepip
	pypy -m pip install --upgrade pip
	pypy -m pip install bottle
