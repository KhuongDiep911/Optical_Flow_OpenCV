#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'run_optical_flow_lucas_kanade'.
Computes the Lucas Kanade optical flow for each data item associated with the
selected games and stores the visualization to disk. The number of results
produced can be limited by specifying the bounds of the loop. Parameters of the
Lucas Kanade optical flow and feature detector are provided in verbose detail
to ease understanding and tuning.
"""

# Optical flow in OpenCV.
# https://docs.opencv.org/3.4.0/d7/d8b/tutorial_py_lucas_kanade.html

# Lucas-Kanade Optical Flow
# https://docs.opencv.org/3.4.0/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323

import os
from jm17290.ops import *
from jm17290.data import Instance
from jm17290.atari.game import Games
from jm17290.optical_flow.ops import *
from jm17290.visualisation import ProgressBar

# Select game.
game = Games.breakout

# Set parameters for feature detection.
# https://docs.opencv.org/3.4.0/dd/d1a/group__imgproc__feature.html#gaaf8a051fb13cab1eba5e2149f75e902f
# mask : Optional region of interest.
# maxCorners : Maximum number of corners to return. If there are more corners than are found, the strongest of them is
# returned. maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned.
# qualityLevel : Parameter characterizing the minimal accepted quality of image corners. The parameter value is
# multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal ) or the Harris
# function response (see cornerHarris ). The corners with the quality measure less than the product are rejected. For
# example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the
# quality measure less than 15 are rejected.
# minDistance: Minimum possible Euclidean distance between the returned corners.
# blockSize: Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.
# See cornerEigenValsAndVecs.
params_feature_detection = dict(maxCorners=50,
                                qualityLevel=0.03,
                                minDistance=1,
                                blockSize=2)

# Set lucas kanade optical flow parameters.
# https://docs.opencv.org/3.4.0/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
# prevImg: First 8-bit input image or pyramid constructed by buildOpticalFlowPyramid.
# nextImg: Second input image or pyramid of the same size and the same type as prevImg.
# prevPts: Vector of 2D points for which the flow needs to be found; point coordinates must be single-precision
# floating-point numbers.
# nextPts: Output vector of 2D points (with single-precision floating-point coordinates) containing the calculated new
# positions of input features in the second image; when OPTFLOW_USE_INITIAL_FLOW flag is passed, the vector must have
# the same size as in the input.
# winSize: Size of the search window at each pyramid level.
# maxLevel: 0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two
# levels are used, and so on; if pyramids are passed to input then algorithm will use as many levels as pyramids have
# but no more than maxLevel.
# criteria: Parameter, specifying the termination criteria of the iterative search algorithm (after the specified
# maximum number of iterations criteria.maxCount or when the search window moves by less than criteria.epsilon.
# Operation flags:
#   OPTFLOW_USE_INITIAL_FLOW uses initial estimations, stored in nextPts; if the flag is not set, then
# prevPts is copied to nextPts and is considered the initial estimate.
#   OPTFLOW_LK_GET_MIN_EIGENVALS use minimum eigen values as an error measure (see minEigThreshold description); if the
# flag is not set, then L1 distance between patches around the original and a moved point, divided by number of pixels
# in a window, is used as a error measure.
params_lucas_kanade = dict(winSize=(3, 3),
                           maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def run_optical_flow_lucas_kanade(game):
    data = os.listdir(game.data)

    progress_bar = ProgressBar(max=len(data), len=20)

    for i, filename in enumerate(data):

        # Instance object capturing the information stored in the name of the file (observation, episode, tick, action).
        data_instance = Instance(filename)

        if data_instance.observation > 1 and data_instance.observation < 1502:
            # Get observation images.
            frames_colour = frames_load_bgr('{0}/{1}'.format(game.data, filename))
            frames_grey_scale = frames_load_grey_scale('{0}/{1}'.format(game.data, filename))

            # Get lucas kanade optical flow.
            vec_start, vec_end = optical_flow_lucas_kanade(frames_grey_scale, params_feature_detection,
                                                           params_lucas_kanade)
            result = optical_flow_lucas_kanade_result(frames_colour[-1], vec_start, vec_end)
            image_save(
                './optical_flow/results/lucas_kanade/{0}/frames/{1}'.format(game.name, data_instance.filename),
                result)

            # Print progress.
            progress_bar.update(i)


run_optical_flow_lucas_kanade(game)
