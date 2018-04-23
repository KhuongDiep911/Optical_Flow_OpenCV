#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'ops.py'.

    Module containing functions which wrap up core optical flow functionality
    from the OpenCV library 'cv2'. Operations are divided into those which
    obtain results and those that create visualisations. The results can
    therefore be obtained for further processing independent of visualisations
    being created.
"""

from jm17290.ops import ColoursNeon, frame_blank

import cv2
import numpy as np


def optical_flow_lucas_kanade(frames, params_f, params_lk):
    """ J.Madge 23.04.2018, 'optical_flow_lucas_kanade'.
    Performs Lucas Kanade optical flow between successive input frames by
    wrapping the OpenCV 'calcOpticalFlowPyrLK' function which employs the
    iterative Lucas Kanade method. See https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    for further details. Start and end positions of all optical flow vectors are
    obtained for each transition between frames and are combined into final
    lists and returned. Vectors which start and end at the same position are
    filtered out to improve clarity. Output of the OpenCV
    'calcOpticalFlowPyrLK' function is described verbosely for ease of
    understanding and tuning.
    :param frames: List of sequential frames on which Lucas Kanade optical
    flow will be performed.
    :param params_f: Dictionary of feature detector parameters.
    :param params_lk: Dictionary of Lucas Kanade optical flow parameters.
    :return: A list containing all the starting positions of the optical flow
    vectors and a list containing all the end positions of the optical flow
    vectors.
    """

    vec_start_lst = np.empty(shape=(0, 2), dtype=np.int)
    vec_end_lst = np.empty(shape=(0, 2), dtype=np.int)

    current_frame = frames[0]
    features_original = cv2.goodFeaturesToTrack(current_frame, mask=None, **params_f)
    for i in range(1, frames.__len__()):

        # Lucas-Kanade, output.
        # https://docs.opencv.org/3.4.0/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
        # newPoints     Output vector of 2D points (with single-precision floating-point coordinates) containing the
        #               calculated new positions of input features in the second image; when OPTFLOW_USE_INITIAL_FLOW
        #               flag is passed, the vector must have the same size as in the input.
        # status        Output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for
        #               the corresponding features has been found, otherwise, it is set to 0.
        # error         Output vector of errors; each element of the vector is set to an error for the corresponding
        #               feature, type of the error measure can be set in flags parameter; if the flow wasn't found then
        #               the error is not defined (use the status parameter to find such cases).
        features_moved, status, error = cv2.calcOpticalFlowPyrLK(current_frame, frames[i], features_original, None,
                                                                 **params_lk)
        features_moved_found = features_moved[status == 1]
        features_original_found = features_original[status == 1]

        for j, (original, moved) in enumerate(zip(features_original_found, features_moved_found)):
            a, b = original.ravel()
            c, d = moved.ravel()

            # Filter out vectors that start and end in the same position.
            if a != c or b != d:
                # Rounding to the nearest pixel.
                vec_start_lst = np.append(vec_start_lst, [[int(round(a)), int(round(b))]], axis=0)
                vec_end_lst = np.append(vec_end_lst, [[int(round(c)), int(round(d))]], axis=0)

        current_frame = frames[i].copy()
        features_original = features_moved_found.reshape(-1, 1, 2)

    return vec_start_lst, vec_end_lst


def optical_flow_lucas_kanade_result(frame_end, vec_start, vec_end):
    """ J.Madge 23.04.2018, 'optical_flow_lucas_kanade_result'.
    Creates a visualisation of the Lucas Kanade optical flow result. Based on
    the implementation found here https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html.
    Draws optical flow vectors that are terminated with a cross on the
    final frame of the sequence on which the Lucas Kanade optical flow
    was calculated.
    :param frame_end: The final frame in the sequence on which Lucas Kanade
    optical flow was performed.
    :param vec_start: List of positions representing the start of optical flow
    vectors.
    :param vec_end: List of positions representing the end of optical flow
    vectors.
    :return: An image visualising the results of Lucas Kanade optical flow
    which is a blend of the final frame of the sequence on which optical flow
    was performed and visualizations of the calculated optical flow vectors.
    """

    mask = frame_end.copy()

    # Draw tracks.
    colour_line = ColoursNeon.neon_blue
    colour_dot = ColoursNeon.neon_pink

    for i, (start, end) in enumerate(zip(vec_start, vec_end)):
        a, b = start.ravel()
        c, d = end.ravel()
        # https: // docs.opencv.org / trunk / dc / da5 / tutorial_py_drawing_functions.html
        mask = cv2.line(mask, (a, b), (c, d), colour_line, 1)
        mask = cv2.circle(mask, (c, d), 1, colour_dot, -1)

    return cv2.addWeighted(frame_end, 0.3, mask, 0.7, 1)


def optical_flow_dense(frames, params_d):
    """ J.Madge 23.04.2018, 'optical_flow_dense'.
    Calculates dense optical flow by wrapping the OpenCV
    'calcOpticalFlowFarneback' function which employs Gunnar Farneback’s
    algorithm. See https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    for further details.
    :param frames: Sequence of frames on which to perform dense optical flow.
    :param params_d: Dictionary of dense optical flow parameters.
    :return: An array representing the calculated 'flow image'.
    """

    result = np.zeros(shape=(84, 84, 2))
    for i in range(0, frames.__len__() - 1):
        flow = cv2.calcOpticalFlowFarneback(prev=frames[i], next=frames[i + 1], **params_d)
        result = np.add(result, flow)
    return result


def optical_flow_dense_result(frame_end, flow):
    """ J.Madge 23.04.2018, 'optical_flow_dense_result'.
    Creates a visualization of the dense optical flow result. Based on the
    implementation found here: https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html.
    :param frame_end: Final frame of the sequence on which dense optical flow
    was calculated.
    :param flow: Calculated optical flow, array of dimensions 84 x 84 x 2.
    :return: An image visualising the results of dense optical flow which is a
    blend of the final frame of the sequence on which optical flow was
    performed and the calculated dense optical flow image.
    """

    hsv = frame_blank()
    hsv[..., 1] = 255  # Set greens to 255.
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.add(bgr, frame_end)


def print_vectors(start, end):
    """ J.Madge 23.04.2018, 'print_vectors'.
    Provides a numeric visualization of the calculated Lucas Kanade optical
    flow vectors by printing them out to console.
    :param start: List of points representing the start of the optical flow
    vectors.
    :param end: List of points representing the end of the optical flow
    vectors.
    :return: None.
    """

    print('#\tStart Vector\tEnd Vector')
    for i in range(0, start.shape[0]):
        print('{0}\t{1}\t\t\t{2}'.format(i, start[i], end[i]))


def print_flow(flow):
    """ J.Madge 23.04.2018, 'print_flow'.
    Provides a numeric visualization of the calculated dense optical flow
    vectors by printing them out to console.
    :param flow: Calculated optical flow image.
    :return: None.
    """

    print('Y\t\tX')
    for i in range(0, flow.shape[0]):
        print('\t\t', end='')
        print('{0}\t\t\t\t'.format(i), end='')
    print()

    for j in range(0, flow.shape[0]):
        print('{0}\t\t'.format(j), end='')
        for k in range(0, flow.shape[1]):
            a = round(flow[j, k][0], 3)
            b = round(flow[j, k][1], 3)
            print('([%.3f],[%.3f])\t\t' % (a, b), end='')
        print()
