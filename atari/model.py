#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'model.py'.

    Source adapted from:
    https://github.com/ppwwyyxx/tensorpack/blob/master/examples/A3C-Gym/train-atari.py#L72

    A constructor is added to the 'Model' class that accepts the number
    of actions for each game as an argument. A reference to the ALE
    (Arcade Learning Environment) is therefore not required.

    The 'get_model' function enables a model to be returned by providing a
    'Game' object which contains information about the game's actions and
    the location of the associated model.
"""

from jm17290.atari.predict import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader

import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc
from tensorpack.tfutils.argscope import argscope
from tensorpack.models.conv2d import Conv2D
from tensorpack.models.pool import MaxPooling
from tensorpack.models.fc import FullyConnected
from tensorpack.models.nonlin import PReLU
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
import tensorpack.tfutils.summary as summary
import tensorpack.tfutils.optimizer as optimizer

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)


class Model(ModelDesc):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def inputs(self):
        assert self.num_actions is not None
        return [tf.placeholder(tf.uint8, (None,) + IMAGE_SHAPE3, 'state'),
                tf.placeholder(tf.int64, (None,), 'action'),
                tf.placeholder(tf.float32, (None,), 'futurereward'),
                tf.placeholder(tf.float32, (None,), 'action_prob'),
                ]

    def _get_NN_prediction(self, image):
        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, activation=tf.nn.relu):
            l = Conv2D('conv0', image, 32, 5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, 32, 5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, 64, 4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, 64, 3)

        l = FullyConnected('fc0', l, 512)
        l = PReLU('prelu', l)
        logits = FullyConnected('fc-pi', l, self.num_actions)  # unnormalized policy
        value = FullyConnected('fc-v', l, 1)
        return logits, value

    def build_graph(self, state, action, futurereward, action_prob):
        logits, value = self._get_NN_prediction(state)
        value = tf.squeeze(value, [1], name='pred_value')  # (B,)
        policy = tf.nn.softmax(logits, name='policy')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(policy + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, self.num_actions), 1)
        advantage = tf.subtract(tf.stop_gradient(value), futurereward, name='advantage')

        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, self.num_actions), 1)  # (B,)
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_sum(policy * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(value - futurereward, name='value_loss')

        pred_reward = tf.reduce_mean(value, name='predict_reward')
        advantage = tf.sqrt(tf.reduce_mean(tf.square(advantage)), name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        cost = tf.truediv(cost, tf.cast(tf.shape(futurereward)[0], tf.float32), name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss,
                                   value_loss, pred_reward, advantage,
                                   cost, tf.reduce_mean(importance, name='importance'))
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


def get_model(game):
    """ J.Madge 23.04.2018, 'get_model'.

    :param game: Instance of the 'Game' class containing information about
    the actions in the game and the location of the associated model.
    :return: A model representing the Atari agents pre-trained to play the
    specified game.
    """
    return OfflinePredictor(PredictConfig(
        model=Model(game.actions),
        session_init=get_model_loader(game.model),
        input_names=['state'],
        output_names=['policy']))
