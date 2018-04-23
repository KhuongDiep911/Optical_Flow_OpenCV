#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'core_operations.py'.

    Class created to provide examples of basic OpenCV Python operation that
    were referred to during the course of this project.
"""

# OpenCV Python tutorials.
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html

# Installing OpenCV for Windows.
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html#install-opencv-python-in-windows

# Reading in an image.
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#basic-ops

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Print the version of OpenCV.
print('OpenCV Version {0}\n'.format(cv2.__version__))

# Image dimensions: 336 x 84.
# Four individual images of size 84 x 84.
file = '../data/ms_pacman/data/2-0-2-0.png'

# Load images.
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
# 'Loads a color image. Any transparency of image will be neglected. It is the default flag.'
load_colour = 1
# 'Loads image in grayscale mode'.
load_gray_scale = 0
# 'Loads image as such including alpha channel'.
load_unchanged = -1

img_color = cv2.imread(file, load_colour)
img_gray_scale = cv2.imread(file, load_gray_scale)
img_unchanged = cv2.imread(file, load_unchanged)

# Display images.
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
cv2.imshow('img_color', img_color)
cv2.imshow('img_gray_scale', img_gray_scale)
cv2.imshow('img_unchanged', img_unchanged)

# Image properties.
# Shape, returns tuple of number of rows, columns and channels.
# Size, total number of pixels in the image.
# Data type.
print('Colour image:  shape({0}), size({1}), data type ({0})'.format(img_color.shape, img_color.size, img_color.dtype))
print('Grey scale image:  shape({0}), size({1}), data type ({0})'.format(img_gray_scale.shape, img_gray_scale.size,
                                                                         img_gray_scale.dtype))
print('Unchanged image:  shape({0}), size({1}), data type ({0})\n'.format(img_unchanged.shape, img_unchanged.size,
                                                                          img_unchanged.dtype))

# Display images with 'matplotlib'.
plt.imshow(img_gray_scale, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# Rows: Zero-based, starting at the top of the image.
pixelRow = 0
# Cols: Zero-based, starting at the left of the image.
pixelCol = 0

# Access individual pixels.
# Pixels return as [ B, G, R ]
blue = 0
green = 1
red = 2

# Method best suited to selecting regions of an array.
print('Inefficient individual pixel access.')
print('Pixel at row({0}), col({1}): {2}'.format(pixelRow, pixelCol, img_color[pixelRow, pixelCol]))
print('Blue: {0}'.format(img_color[pixelRow, pixelCol, blue]))
print('Green: {0}'.format(img_color[pixelRow, pixelCol, green]))
print('Red: {0}\n'.format(img_color[pixelRow, pixelCol, red]))

# Method best suited to individual pixel access.
print('Efficient individual pixel access using Numpy.')
print('Blue: {0}'.format(img_color.item(pixelRow, pixelCol, blue)))
print('Green: {0}'.format(img_color.item(pixelRow, pixelCol, green)))
print('Red: {0}\n'.format(img_color.item(pixelRow, pixelCol, red)))

# Modifying pixels.
print('Inefficient pixel modification.')
print('Original pixel: {0}'.format(img_color[pixelRow, pixelCol]))
img_color[pixelRow, pixelCol] = [255, 255, 255]
print('Modified pixel: {0}\n'.format(img_color[pixelRow, pixelCol]))

print('Efficient pixel modification using Numpy.')
print('Original pixel: {0} {1} {2}'.format(img_color.item(pixelRow, pixelCol, blue),
                                           img_color.item(pixelRow, pixelCol, green),
                                           img_color.item(pixelRow, pixelCol, red)))
img_color.itemset((pixelRow, pixelCol, blue), 10)
img_color.itemset((pixelRow, pixelCol, green), 20)
img_color.itemset((pixelRow, pixelCol, red), 30)
print('Modified pixel: {0} {1} {2}'.format(img_color.item(pixelRow, pixelCol, blue),
                                           img_color.item(pixelRow, pixelCol, green),
                                           img_color.item(pixelRow, pixelCol, red)))

# Region of interest (ROI).
frame_rows = 84
frame_cols = 84
frame1 = img_color[0:frame_rows, (frame_cols * 0):(frame_cols * 1)]
frame2 = img_color[0:frame_rows, (frame_cols * 1):(frame_cols * 2)]
frame3 = img_color[0:frame_rows, (frame_cols * 2):(frame_cols * 3)]
frame4 = img_color[0:frame_rows, (frame_cols * 3):(frame_cols * 4)]
cv2.imwrite("./core_operations_output/frame1.png", frame1)
cv2.imwrite("./core_operations_output/frame2.png", frame2)
cv2.imwrite("./core_operations_output/frame3.png", frame3)
cv2.imwrite("./core_operations_output/frame4.png", frame4)

# Image addition.
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
# OpenCV function. 'OpenCV addition is a saturated operation ... [and] will provide better results.'
frames_one_four_add_opencv = cv2.add(frame1, frame4)
cv2.imwrite('./core_operations_output/frames_one_four_add_opencv.png', frames_one_four_add_opencv)
# Numpy function. 'Numpy addition is a modulo operation'.
frames_one_four_add_numpy = frame1 + frame4
cv2.imwrite('./core_operations_output/frames_one_four_add_numpy.png', frames_one_four_add_numpy)

# Image blending.
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
frames_one_four_blend = cv2.addWeighted(frame1, 0.5, frame4, 0.5, 0)
cv2.imwrite('./core_operations_output/frames_one_four_blend.png', frames_one_four_blend)

# Splitting and merging image channels.
# 'The B,G,R channels of an image can be split into their individual planes when needed.
# Then, the individual channels can be merged back together to form a BGR image again.'
# Splitting channels.
img_color_b, img_color_g, img_color_r = cv2.split(img_color)
# Or.
# img_color_b = img_color[:, :, blue]
# img_color_g = img_color[:, :, green]
# img_color_r = img_color[:, :, red]
cv2.imshow('img_color_b', img_color_b)
cv2.imshow('img_color_g', img_color_g)
cv2.imshow('img_color_r', img_color_r)
# Merging channels.
img_color_merge = cv2.merge((img_color_b, img_color_g, img_color_r))
cv2.imshow('img_color_merge', img_color_merge)

# Modifying individual image channels.
img_color[:, :, red] = 0
cv2.imshow('img_no_red', img_color)

# Saving an image.
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
img_save = './core_operations_output/mod.png'
cv2.imwrite(img_save, img_color)

# Close all open images windows upon key press.
cv2.waitKey(0)
cv2.destroyAllWindows()
