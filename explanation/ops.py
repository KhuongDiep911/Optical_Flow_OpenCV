#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'ops.py'.

    Module containing several functions which wrap up the functionality of
    LIME. Links listed at the start of the file serve as a points of reference
    to related material.
"""

# LIME (Local Interpretable Model-Agnostic Explanations)

# 'Why Should I Trust You?' Explaining the Predictions of Any Classifier.
# https://arxiv.org/pdf/1602.04938.pdf

# Anchors: High-Precision Model-Agnostic Explanations
# https://homes.cs.washington.edu/~marcotcr/aaai18.pdf

# Explanation by O'Reilly
# https://www.oreilly.com/learning/introduction-to-local-interpretable-model-agnostic-explanations-lime

# GitHub Repository
# https://github.com/marcotcr/lime

# LIME documentation, lime_image.LimeImageExplainer().explain_instance, 'classifier_fn' parameter is:
# 'classifier prediction probability function, which takes a numpy array and outputs prediction probabilities'.
# http://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image

# TensorPack OfflinePredictor that needs to be adapted.
# https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/predict/base.py

# OfflinePredictor instantiated in 'train-atari.py'.
# https://github.com/ppwwyyxx/tensorpack/blob/master/examples/A3C-Gym/train-atari.py#L290

# OfflinePredictor used here with 'argmax' to obtain an action.
# https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DeepQNetwork/common.py

from jm17290.ops import *
from jm17290.data import Instance
from jm17290.atari.model import get_model
from jm17290.visualisation import ProgressBar

import os
import random

from lime import lime_image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


def explain_random(game, explainations):
    """ J.Madge 23.04.2018, 'explain_random'.
    Explains the action taken by the Atari agent for a random data item of the
    specified games and saves the result.
    :param game: The game for which a random data item is to be retrieved
    and explained.
    :param explainations: The number of random data samples that are
    to be explained for the specified game.
    :return: None.
    """

    model = get_model(game)
    data = os.listdir(game.data)

    progress_bar = ProgressBar(max=explainations, len=20)

    for i in range(explainations):
        # Get random observation file.
        filename = random.choice(data)

        # Instance object capturing the information stored in the name of the file (observation, episode, tick, action).
        data_instance = Instance(filename)

        # Get observation image.
        observation = image_load_rgb('{0}/{1}'.format(game.data, filename))

        # Get action.
        action = model(observation).argmax()

        # Get explanation.
        image, mask = explain(action, observation, model, list(range(game.actions)))

        # Convert image to rgb.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # De-normalise.
        marked = mark_boundaries(image, mask) * 255

        # Save image.
        cv2.imwrite('./explanation/results/{0}/frames/{1}'.format(game.name, data_instance.filename), marked)

        # Print progress.
        progress_bar.update(i)


def explain_show(filename, game):
    """ J.Madge 23.04.2018, 'explain_show'.
    Explains the action taken by the agent for the specified file and displays
    the resulting LIME interpretation.
    :param filename: Path to the data item which is to be explained.
    :param game: Game to which the specified data item belongs.
    :return: None
    """

    model = get_model(game)
    data_instance = Instance(filename)
    observation = image_load_rgb('{0}/{1}'.format(game.data, data_instance.filename))

    temp, mask = explain(data_instance, observation, model, game)

    # Show the explanation.
    # http://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.mark_boundaries
    # 'Return image with boundaries between labeled regions highlighted.'
    plt.imshow(mark_boundaries(temp, mask))
    plt.title('Data File: {0}, Action: {1}'.format(data_instance.filename, data_instance.action))
    plt.show()


def explain_save(filename, game):
    """ J.Madge 23.04.2018, 'explain_save'.
    Explains the action taken by the agent for the specified file and displays
    the resulting LIME interpretation.
    :param filename: Path to the data item which is to be explained.
    :param game: Game to which the specified data item belongs.
    :return: None
    """

    model = get_model(game)
    data_instance = Instance(filename)
    observation = image_load_rgb('{0}/{1}'.format(game.data, data_instance.filename))

    # Get explanation.
    image, mask = explain(data_instance, observation, model, game)
    # Denormalise.
    marked = mark_boundaries(image, mask) * 255
    # Save image.
    cv2.imwrite('./lime/results/{0}/{1}'.format(game.name, data_instance.filename), marked)


def explain(action, observation, classification_func, actions):
    """ J.Madge 23.04.2018, 'explain'.
    Performs explanation of the provided action and observation by wrapping
    LIME functionality. Parameters of the LIME functions are provided in
    verbose style to assist understanding and tuning.
    :param action: The action taken by the agent based on the provided
    observation that is to be explained.
    :param observation: The observation on which the provided action was taken.
    :param classification_func: The function used to obtain the action from
    the observation.
    :param actions: The actions available to the agent for the given game.
    :return: The image used in the explanation and a mask which identifies
    regions of the image that contribute towards the specified action being
    taken.
    """

    # http://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.LimeImageExplainer
    # kernel_width          Kernel width for the exponential kernel.
    # verbose               If true, print local prediction values from linear model.
    # feature_selection     Feature selection method. can be ‘forward_selection’, ‘lasso_path’, ‘none’ or ‘auto’. See
    #                       function ‘explain_instance_with_data’ in lime_base.py for details on what each of the
    #                       options does.
    explainer = lime_image.LimeImageExplainer()  # http://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.LimeImageExplainer.explain_instance
    # data_row          1d numpy array, corresponding to a row #TODO J.Madge 20.04.2018 Is this doc correct?
    # classifier_fn     Classifier prediction probability function, which takes a numpy array and outputs prediction
    #                   probabilities. For ScikitClassifiers , this is classifier.predict_proba.
    # labels            Iterable with labels to be explained.
    # top_labels        If not None, ignore labels and produce explanations for the K labels with highest prediction
    #                   probabilities, where K is this parameter.
    # num_features      Maximum number of features present in explanation. Default=100000
    # num_samples       Size of the neighborhood to learn the linear model. Default=1000
    # batch_size        Default=10
    # qs_kernel_size    Default=4
    # distance_metric   The distance metric to use for weights. Default='cosine'
    # model_regressor   sklearn regressor to use in explanation. Defaults Ridge regression in LimeBase. Must have
    #                   model_regressor.coef (to) – 'sample_weight' as a parameter to model_regressor.fit() (and) –
    #                   qs_kernel_size – the size of the kernal to use for the quickshift segmentation
    # TODO J.Madge Implementation of the 'atari_model' prediction function limits the batch size to 1.
    # TODO J.Madge Sometimes error since pre-obtained classifications do not match new classifications.
    explanation = explainer.explain_instance(observation, classification_func, labels=actions, hide_color=0,
                                             top_labels=5,
                                             num_features=100000, num_samples=1000, batch_size=1,
                                             distance_metric='cosine',
                                             model_regressor=None)

    # http://lime-ml.readthedocs.io/en/latest/lime.html#lime.lime_image.ImageExplanation.get_image_and_mask
    # label             Label to explain.
    # positive_only     If True, only take super-pixels that contribute to the prediction of the label. Otherwise, use
    #                   the top num_features super-pixels, which can be positive or negative towards the label.
    # hide_rest         If True, make the non-explanation part of the return image gray.
    # num_features      Number of super-pixels to include in explanation.
    return explanation.get_image_and_mask(action, positive_only=False, hide_rest=False, num_features=3)
