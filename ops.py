#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'ops.py'.

    Module containing basic file IO and manipulation operations. Many of the
    functions wrap basic functionality of the OpenCV Python library 'cv2' and
    Numpy. These functions are used throughout the project to provide
    consistency and avoid duplication.
"""

import cv2
import random
import numpy as np
from enum import IntEnum


class LoadType(IntEnum):
    """ J.Madge 23.04.2018, 'LoadType'.
    Wrapper for OpenCV identifiers which indicate how the specified image is to
    be loaded.
    """

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
    # 'Loads a color image. Any transparency of image will be neglected. It is the default flag.'
    COLOUR = 1
    # 'Loads image in grayscale mode'.
    GREY_SCALE = 0
    # 'Loads image as such including alpha channel'.
    UNCHANGED = -1


class ColoursNeon:
    """ J.Madge 23.04.2018, 'ColoursNeon'.
    Colours specifically selected to be visible when overlaid on existing
    images. Used extensively in this project when visualising results.
    """

    # BGR format.
    neon_pink = [143, 89, 255]
    neon_orange = [94, 138, 253]
    neon_yellow = [0, 227, 224]
    neon_blue = [221, 221, 1]
    neon_green = [175, 191, 0]


class Frame:
    """ J.Madge 23.04.2018, 'Frame'.
    Holds information about images that represent each data instance. Ensures
    these values are not repeated throughout the program and limits the
    'special numbers' in the code.
    """

    frame_count = 4
    frame_rows = 84
    frame_cols = 84
    frame_depth = 3


def opencv_version():
    """ J.Madge 23.04.2018, 'opencv_version'.
    Prints the version of OpenCV used in the project.
    :return: None.
    """

    print('OpenCV Version {0}\n'.format(cv2.__version__))


def image_load_bgr(path):
    """ J.Madge 23.04.2018, 'image_load_bgr'.
    Loads the image at the specified path in the BGR colour space which is the
    OpenCV default.
    :param path: Path to the image to be loaded.
    :return: Colour image with BGR colour space.
    """

    return cv2.imread(path, LoadType.COLOUR)


def image_load_rgb(path):
    """ J.Madge 23.04.2018, 'image_load_rgb'.
    Loads the image at the specified path in the RGB colour space which is not
    the OpenCV default, hence a conversion is performed.
    :param path: Path to the image to be loaded.
    :return: Colour image with RGB colour space.
    """

    return cv2.cvtColor(cv2.imread(path, LoadType.COLOUR), cv2.COLOR_BGR2RGB)


def image_load_hsv(path):
    """ J.Madge 23.04.2018, 'image_load_hsv'.
    Loads the image at the specified path in the HSV colour space which is not
    the OpenCV default, hence a conversion is performed.
    :param path: Path to the image to be loaded.
    :return: Colour image with HSV colour space.
    """

    return cv2.cvtColor(cv2.imread(path, LoadType.COLOUR), cv2.COLOR_BGR2HSV)


def image_load_grey_scale(path):
    """ J.Madge 23.04.2018, 'image_load_grey_scale'.
    Loads the image at the specified path as grey scale.
    :param path: Path to the image to be loaded.
    :return: Grey scale image.
    """

    return cv2.imread(path, LoadType.GREY_SCALE)


def frames(image):
    """ J.Madge 23.04.2018, 'frames'.
    Separates the provide image into its constituent frames and returns them as
    a list.
    :param image: Image to be separate into it's constituent frames. The image
    must have dimensions 84 x 336 x 3 which is consistent with the size of the
    data instances associated with this project.
    :return: List from 84 x 84 x 3 frames that comprised the specified image.
    """

    frames = []

    for i in range(0, Frame.frame_count):
        frames.append(image[0:Frame.frame_rows, (Frame.frame_cols * i):(Frame.frame_cols * (i + 1))])
    return frames


def frames_load_rgb(path):
    """ J.Madge 23.04.2018, 'frames_load_rgb'.
    Convenience function, loads frames from image in the RGB colour space.
    :param path: Path to the image to be loaded and partitioned into frames.
    :return: List from 84 x 84 x 3 frames in the RGB colour space that
    comprised the specified image.
    """

    return frames(image_load_rgb(path))


def frames_load_bgr(path):
    """ J.Madge 23.04.2018, 'frames_load_bgr'.
    Convenience function, loads frames from image in the BGR colour space.
    :param path: Path to the image to be loaded and partitioned into frames.
    :return: List from 84 x 84 x 3 frames in the BGR colour space that
    comprised the specified image.
    """

    return frames(image_load_bgr(path))


def frames_load_grey_scale(path):
    """ J.Madge 23.04.2018, 'frames_load_grey_scale'.
    Convenience function, loads frames from grey scale image.
    :param path: Path to the image to be loaded and partitioned into frames.
    :return: List from 84 x 84 x 3 grey scale frames that comprised the
    specified image.
    """

    return frames(image_load_grey_scale(path))


def frames_stack(frames):
    """ J.Madge 23.04.2018, 'frames_stack'.
    Stacks a list of four 84 x 84 x 3 frames into a array of dimensions
    84 x 84 x 12.
    :param frames: A list of four frames of dimensions 84 x 84 x 3.
    :return: An array of frames stacked depth-wise (along the third axis) of
    dimensions 84 x 84 x 12. See https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.dstack.html
    for further details.
    """

    return np.dstack((frames[0], frames[1], frames[2], frames[3]))


def frames_stack_rgb(path):
    """ J.Madge 23.04.2018, 'frames_stack_rgb'.
    Convenience function, loads an image in the RGB colour space, separates it
    into frames and stacks them into an array of dimensions 84 x 84 x 12.
    :param path: Path to the image to be loaded.
    :return: Array of dimensions 84 x 84 x 12 consisting of RGB frames of the
    original image stacked depth-wise (along the third axis).
    """

    return frames_stack(frames_load_rgb(path))


def frames_stack_bgr(path):
    """ J.Madge 23.04.2018, 'frames_stack_bgr'.
    Convenience function, loads an image in the BGR colour space, separates it
    into frames and stacks them into an array of dimensions 84 x 84 x 12.
    :param path: Path to the image to be loaded.
    :return: Array of dimensions 84 x 84 x 12 consisting of BGR frames of the
    original image stacked depth-wise (along the third axis).
    """

    return frames_stack(frames_load_bgr(path))


def image_show(title, image, delay=0):
    """ J.Madge 23.04.2018, 'image_show'.
    Displays an image in a titled widow for a defined period of time.
    :param title: Title of the window in which the image is displayed.
    :param image: Image to be displayed in the window.
    :param delay: The time (in milliseconds) for which the image will be
    displayed.
    :return: None.
    """

    cv2.imshow(title, image)
    cv2.waitKey(delay)


def image_show_multi(title, images, delay=0):
    """ J.Madge 23.04.2018, 'image_show_multi'.
    Displays a list of images one after the other in a titled window separated
    by a specified period of time.
    :param title: Title of the window in which the images is displayed.
    :param images: List of images to be displayed sequentially.
    :param delay: Delay (in milliseconds between the presentation of each
    image.
    :return: None.
    """

    for i in images:
        image_show(title, i, delay)
        cv2.waitKey(delay)


def image_save(path, image):
    """ J.Madge 23.04.2018, 'image_save'.
    Save an image to a specified location.
    :param path: Path to which the image will be saved.
    :param image: Image to save.
    :return: None.
    """

    cv2.imwrite(path, image)


def frame_blank():
    """ J.Madge 23.04.2018, 'frame_blank'.
    Gets a blank (empty) array with the same dimensions as frames in the data.
    :return: A blank (empty) array of dimensions 84 x 84 x 3 which match those
    of the frames which comprise the data.
    """

    return np.zeros((Frame.frame_rows, Frame.frame_cols, Frame.frame_depth), dtype=np.uint8)


def colour_random():
    """ J.Madge 23.04.2018, 'colour_random'.
    Selects a random colour from the 'ColoursNeon' class.
    :return: A random colour from the 'ColoursNeon' class.
    """

    return random.choice([ColoursNeon.neon_pink, ColoursNeon.neon_orange, ColoursNeon.neon_yellow,
                          ColoursNeon.neon_blue, ColoursNeon.neon_green])
