#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'inception_v3.py'.

    Script based on the 'Image Classification with Keras' sample code provided
    in the LIME GitHub repository.
    https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20Image%20Classification%20Keras.ipynb

    The script was used to become familiar with LIME and was referred to during
    the project but was not directly used in any analysis.
"""

### Imports.
from jm17290.ops import *

import numpy as np

import matplotlib.pyplot as plt

import keras
from keras.preprocessing import image
from keras.applications import inception_v3 as inc_net
from keras.applications.imagenet_utils import decode_predictions

from lime import lime_image

from skimage.segmentation import mark_boundaries

### Read images.
opencv_version()
# Load game.
# game = 'asteroids'
game = 'battle_zone'
# game = 'breakout'
# game = 'gopher'
# game = 'james_bond'
# game = 'ms_pacman'
# game = 'road_runner'
# game = 'tennis'

file = './data/{0}.png'.format(game)

# frames_colour = frames_load(file, LoadType.COLOUR)
# frames_grey_scale = frames_load(file, LoadType.GREY_SCALE)
# image_show_multi('Debug', frames_grey_scale, 500)
# image_show_multi('Debug', frames_colour, 500)

# Save individual frames so that they can be loaded into keras.
# for x, f in enumerate(frames_colour):
#     cv2.imwrite('./data/{0}/{1}.png'.format(game, x), f)

### Get Karas InceptionV3 image predictions.
print('Keras Version {0}'.format(keras.__version__))

inet_model = inc_net.InceptionV3()


def transform_img_fn(path_list):
    '''Returns a 4D numpy array of stacked images where given shape(a, b, c, d),
    a - image identifier,
    b - y dimension,
    c - x dimension,
    d - Images values (RGB), normalized between -1 and 1, ((2(x/255))-1).'''

    out = []
    for img_path in path_list:
        # Load an image into PIL format.
        # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/load_img
        img = image.load_img(img_path)

        # Converts a PIL Image instance to a Numpy array
        # https: // www.tensorflow.org / api_docs / python / tf / keras / preprocessing / image / img_to_array
        x = image.img_to_array(img)

        # 'Insert a new axis that will appear at the axis position in the expanded array shape'.
        # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.expand_dims.html
        x = np.expand_dims(x, axis=0)

        # 'Preprocesses a tensor or Numpy array encoding a batch of images.'
        # https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L149
        x = inc_net.preprocess_input(x)

        out.append(x)
    return np.vstack(out)


images = transform_img_fn([file])

# Changes normalized RBG value space from (-1 -> 1) to (0 -> 1).
# plt.imshow(images[0] / 2 + 0.5)
# Blocks execution of the script.
# plt.show()

# https://keras.io/models/model/#predict
# x             The input data, as a Numpy array (or list of Numpy arrays if the model has multiple outputs).
# batch_size    Integer. If unspecified, it will default to 32.
# verbose       Verbosity mode, 0 or 1.
# steps         Total number of steps (batches of samples) before declaring the prediction round finished. Ignored with the default value of None.
# Returns (1, 1000) ndarray of predictions.
preds = inet_model.predict(images)

# https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L181
# preds         Numpy tensor encoding a batch of predictions.
# top           Integer, how many top-guesses to return (default, 5).
# Returns A list of lists of top class prediction tuples `(class_name, class_description, score)`. One list of tuples
# per sample in batch input.
for x in decode_predictions(preds)[0]:
    print(x)

### Get lime prediction.
# http://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.LimeImageExplainer
explainer = lime_image.LimeImageExplainer()

# Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels

# http://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.LimeImageExplainer.explain_instance
# data_row              1d numpy array, corresponding to a row
# classifier_fn         classifier prediction probability function, which takes a numpy array and outputs prediction
#                       probabilities. For ScikitClassifiers , this is classifier.predict_proba.
# labels                iterable with labels to be explained.
# top_labels            if not None, ignore labels and produce explanations for the K labels with highest prediction
#                       probabilities, where K is this parameter.
# num_features          maximum number of features present in explanation
# num_samples           size of the neighborhood to learn the linear model
# distance_metric       the distance metric to use for weights.
# model_regressor       sklearn regressor to use in explanation. Defaults
#                       Ridge regression in LimeBase. Must have model_regressor.coef (to) –
#                       'sample_weight' as a parameter to model_regressor.fit() (and) –
#                       qs_kernel_size – the size of the kernal to use for the quickshift segmentation
explanation = explainer.explain_instance(images[0], inet_model.predict, top_labels=5, hide_color=0, num_samples=1000)

# http://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.ImageExplanation.get_image_and_mask
# label                 Label to explain (see, https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).
# positive_only         If True, only take superpixels that contribute to the prediction of the label. Otherwise, use
#                       the top num_features superpixels, which can be positive or negative towards the label.
# hide_rest             If True, make the non-explanation part of the return image gray.
# num_features          Number of superpixels to include in explanation.
temp, mask = explanation.get_image_and_mask(530, positive_only=True, num_features=5, hide_rest=True)

# Show the explanation.
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()
