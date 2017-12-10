Thoughts
========

1. Implement MCTS with Dagger Agent and see if it is good
1. Implement Policy Gradient
1. Train same network on Existing dataset and compete with Dagger Agent

Questions
=========

1. Is the CNN confused? Do we really need two networks?
1. Purely Supervised, Purely DAgger or Combined, which one is better?

Discoveries
===========
$ python3 dual_vs_mcts.py treesup treesup -n 10 -s
ARENA: DUAL treesup-8000 VERSES MCTS treesup-8000
match 0 winner b-treesup [14.3932 sec]
match 1 winner b-treesup [20.4719 sec]
match 2 winner b-treesup [18.8090 sec]
match 3 winner b-treesup [27.0037 sec]
match 4 winner b-treesup [23.7417 sec]
match 5 winner a-treesup [42.7777 sec]
match 6 winner b-treesup [13.9471 sec]
match 7 winner b-treesup [15.1900 sec]
match 8 winner b-treesup [37.5196 sec]
match 9 winner b-treesup [14.2691 sec]
Of the 10 games between them
  treesup as black wins 0
  treesup as white wins 1
  treesup overall wins 1
  treesup against treesup overall win-rate 0.100000

