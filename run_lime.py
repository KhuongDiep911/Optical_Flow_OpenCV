#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'run_lime'.
Obtains LIME image explanations for a random sample data items associated with
each game and saves them to disk. This method ensure a broad set of
explanations are obtained despite limited computing resources.
"""

from jm17290.explanation.ops import *
from jm17290.atari.game import Games

games = [Games.asteroids,
         Games.battle_zone,
         Games.breakout,
         Games.gopher,
         Games.james_bond,
         Games.ms_pacman,
         Games.road_runner,
         Games.tennis]

explanations_per_game = 1

for game in games:
    explain_random(game, explanations_per_game)
