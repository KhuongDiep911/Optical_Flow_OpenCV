#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'run_optical_flow_dense'.
Computes the dense optical flow for each data item associated with the selected
games and stores the visualization to disk. The number of results produced can
be limited by specifying the bounds of the loop. Parameters of the dense
optical flow are provided in verbose detail to ease understanding and tuning.
"""

# Optical flow in OpenCV.
# https://docs.opencv.org/3.4.0/d7/d8b/tutorial_py_lucas_kanade.html

# Dense optical flow in OpenCV.
# https://docs.opencv.org/3.4.0/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af

import os
from jm17290.ops import *
from jm17290.data import Instance
from jm17290.atari.game import Games
from jm17290.optical_flow.ops import *
from jm17290.visualisation import ProgressBar

# Select game.
game = Games.battle_zone

# Set parameters for dense optical flow.
# https://docs.opencv.org/3.4.0/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af
# prev : First 8-bit single-channel input image.
# next : Second input image of the same size and the same type as prev.
# flow : Computed flow image that has the same size as prev and type CV_32FC2.
# pyr_scale : Parameter specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a
# classical pyramid, where each next layer is twice smaller than the previous one.
# levels : Number of pyramid layers including the initial image; levels=1 means that no extra layers are created and
# only the original images are used.
# winsize : Averaging window size; larger values increase the algorithm robustness to image noise and give more chances
# for fast motion detection, but yield more blurred motion field.
# iterations : Number of iterations the algorithm does at each pyramid level.
# poly_n : Size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the
# image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field,
# typically poly_n =5 or 7.
# poly_sigma : Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial
# expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
# flags : Operation flags that can be a combination of the following:
#   OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
#   OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize×winsize filter instead of a box filter of the same size for
#       optical flow estimation; usually, this option gives z more accurate flow than with a box filter, at the cost of
#       lower speed; normally, winsize for a Gaussian window should be set to a larger value to achieve the same level
#       of robustness.
params_dense_optical_flow = dict(flow=None, pyr_scale=0.5, levels=2, winsize=6, iterations=3, poly_n=7, poly_sigma=1.5,
                                 flags=0)


# Get dense optical flow for each captured images of the selected game.
def run_optical_flow_dense(game):
    data = os.listdir(game.data)

    progress_bar = ProgressBar(max=len(data), len=20)

    for i, filename in enumerate(data):

        # Instance object capturing the information stored in the name of the file (observation, episode, tick, action).
        data_instance = Instance(filename)

        if data_instance.observation < 1500:
            # Get observation images.
            frames_colour = frames_load_bgr('{0}/{1}'.format(game.data, filename))
            frames_grey_scale = frames_load_grey_scale('{0}/{1}'.format(game.data, filename))

            # Get dense optical flow.
            flow = optical_flow_dense(frames_grey_scale, params_dense_optical_flow)
            result = optical_flow_dense_result(frames_colour[-1], flow)
            image_save('./optical_flow/results/dense/{0}/{1}'.format(game.name, data_instance.filename), result)

            # Print progress.
            progress_bar.update(i)


run_optical_flow_dense(game)
