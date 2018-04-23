#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'run_classification_verification'.
Validates the classification of the captured data by comparing against new
classifications by the loaded model.
"""

# Imports.
import os
from jm17290.ops import *
from jm17290.atari.game import Games
from jm17290.atari.model import get_model
from jm17290.data import Instance
from jm17290.visualisation import ProgressBar

# Select game.
game = Games.battle_zone

# Load Atari model.
atari_model = get_model(game)


# Get predictions using loaded model.
def classification_verification(game, atari_model):
    data_item_count = 0
    prediction_matches = 0

    data = os.listdir(game.data)

    progress_bar = ProgressBar(max=len(data), len=20)

    for i, filename in enumerate(data):
        # Instance object capturing the information stored in the name of the file (observation, episode, tick, action).
        data_instance = Instance(filename)

        # Get observation image.
        ob = image_load_rgb('{0}/{1}'.format(game.data, filename))

        # Get predicted action.
        act = atari_model(ob).argmax()

        # Keep track of matching predictions.
        data_item_count += 1
        if act == data_instance.action:
            prediction_matches += 1

        # Print progress.
        progress_bar.update(i)

    print('\nResult, {0}: {1}/{2}'.format(game.name, prediction_matches, data_item_count))


classification_verification(game, atari_model)
