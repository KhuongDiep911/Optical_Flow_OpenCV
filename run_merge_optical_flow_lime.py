#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'run_merge_optical_flow_lime'.
Merge output of dense optical flow and LIME to obtain new insights. For each
LIME explanation, if the corresponding dense optical flow visualization exists,
the two images are blended to form a new insight.
"""

import os
from jm17290.ops import *
from jm17290.atari.game import Games

games = [Games.asteroids,
         Games.battle_zone,
         Games.breakout,
         Games.gopher,
         Games.james_bond,
         Games.ms_pacman,
         Games.road_runner,
         Games.tennis]

for game in games:
    # Get the LIME results for each game.
    data = os.listdir('./explanation/results/{0}/frames'.format(game.name))
    for file in data:
        lime_path = './explanation/results/{0}/frames/{1}'.format(game.name, file)
        optical_flow_path = './optical_flow/results/dense/{0}/frames/{1}'.format(game.name, file)
        if os.path.isfile(optical_flow_path):
            img_optical_flow = image_load_bgr(optical_flow_path)
            img_lime = image_load_bgr(lime_path)

            # Create new (84 x 336 x 3) image by stacking repeated optical flow images in the y-axis.
            img_optical_flow_stacked = np.hstack(
                (img_optical_flow, img_optical_flow, img_optical_flow, img_optical_flow))

            # Save overlay of LIME image and optical flow image.
            result = cv2.addWeighted(img_lime, 0.5, img_optical_flow_stacked, 0.5, 1)

            print('File: {0}, Stacked shape: {1}'.format(file, img_optical_flow_stacked.shape))
            cv2.imwrite('./merged_lime_optical_flow/{0}'.format(file), result)
