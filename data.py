#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'data.py'.

    The 'Instance' class extract pre-obtained information from the filename
    of existing data items.
"""

class Instance:
    """ J.Madge 23.04.2018, 'Instance'.
    Strips and holds information about a data instance from the provided filename.

    Observation     Incremented observation number ranging from 0->4999 for the
                    current data set.
    Episode         Incremented gamed number, < 0 if a new game started during data capture.
    Tick            Observation number of the current episode.
    Action          The resulting action taken by the agent given the input.
    """

    def __init__(self, filename):
        """ J.Madge 23.04.2018, '__init__'.
        Initializes an 'Instance' object.
        :param filename: Filename of the data instance with format:
        '[observation]-[episode]-[tick]-[action].png'.
        """

        self.filename = filename

        # Remove '.png' from file name and split into components.
        n = filename[:-4].split('-')

        self.observation = int(n[0])
        self.episode = int(n[1])
        self.tick = int(n[2])
        self.action = int(n[3])
