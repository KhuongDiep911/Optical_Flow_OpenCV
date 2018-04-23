#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'game.py'.
    Classes to hold information about Atari games.
"""

class Game:
    """ J.Madge 23.04.2018, 'Game'.

    Holds information about Atari games.

    Name, friendly game identifier.
    Actions, number of actions that can be
    taken when playing the game.
    Model, local location of the model trained to play the game.
    Data, location of the captured data associated with the game.
    """

    def __init__(self, name, actions, model, data):
        self.name = name
        self.actions = actions
        self.model = model
        self.data = data


class Games:
    """ J.Madge 23.04.2018, 'Games'.

    Stores information about Atari games used in this study. The provided links
    indicate the action associated with each integer value for each game.
    """

    # https://github.com/mgbellemare/Arcade-Learning-Environment/blob/ed5115bc29d4b264519e7fe7a455794ecc6ba42b/src/games/supported/Asteroids.cpp#L84
    asteroids = Game(name='asteroids', actions=14, model='./atari/models/Asteroids-v0.tfmodel',
                     data='./atari/data/asteroids/data')

    # https://github.com/mgbellemare/Arcade-Learning-Environment/blob/ed5115bc29d4b264519e7fe7a455794ecc6ba42b/src/games/supported/BattleZone.cpp#L91
    battle_zone = Game(name='battle_zone', actions=18, model='./atari/models/BattleZone-v0.tfmodel',
                       data='./atari/data/battle_zone/data')

    # https://github.com/mgbellemare/Arcade-Learning-Environment/blob/ed5115bc29d4b264519e7fe7a455794ecc6ba42b/src/games/supported/Breakout.cpp#L79
    breakout = Game(name='breakout', actions=4, model='./atari/models/Breakout-v0.npy', data='./atari/data/breakout/data')

    # https://github.com/mgbellemare/Arcade-Learning-Environment/blob/ed5115bc29d4b264519e7fe7a455794ecc6ba42b/src/games/supported/Gopher.cpp#L80
    gopher = Game(name='gopher', actions=8, model='./atari/models/Gopher-v0.tfmodel', data='./atari/data/gopher/data')

    # https://github.com/mgbellemare/Arcade-Learning-Environment/blob/ed5115bc29d4b264519e7fe7a455794ecc6ba42b/src/games/supported/JamesBond.cpp#L82
    james_bond = Game(name='james_bond', actions=18, model='./atari/models/Jamesbond-v0.tfmodel',
                      data='./atari/data/james_bond/data')

    # https://github.com/mgbellemare/Arcade-Learning-Environment/blob/ed5115bc29d4b264519e7fe7a455794ecc6ba42b/src/games/supported/MsPacman.cpp#L80
    ms_pacman = Game(name='ms_pacman', actions=9, model='./atari/models/MsPacman-v0.tfmodel',
                     data='./atari/data/ms_pacman/data')

    # https://github.com/mgbellemare/Arcade-Learning-Environment/blob/ed5115bc29d4b264519e7fe7a455794ecc6ba42b/src/games/supported/RoadRunner.cpp#L89
    road_runner = Game(name='road_runner', actions=18, model='./atari/models/RoadRunner-v0.tfmodel',
                       data='./atari/data/road_runner/data')

    # https://github.com/mgbellemare/Arcade-Learning-Environment/blob/ed5115bc29d4b264519e7fe7a455794ecc6ba42b/src/games/supported/Tennis.cpp#L76
    tennis = Game(name='tennis', actions=18, model='./atari/models/Tennis-v0.tfmodel', data='./atari/data/tennis/data')