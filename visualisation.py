#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018, 'visualisation.py'.
Reusable visualisation classes and methods.
"""

import sys


class ProgressBar:
    """ J.Madge 24.04.2018, 'ProgressBar'.
    Progress bar printed in the console to monitor the progress of long-running
    operations.
    """

    def __init__(self, max, len):
        """ J.Madge 24.04.2018, '__init__'.
        Initialises the progress bar.
        :param max: Value which indicates when the progress bar is
        full/complete.
        :param len: The length of the progress bar as presented in the console.
        """

        self.max = max
        self.len = len

    def update(self, i):
        """ J.Madge 24.04.2018, 'update'.
        Updates the progress bar and causes it to be printed in the console.
        :param i: Integer value less than or equal to the progress bar's
        maximum value.
        :return: None.
        """

        print('\r', end='')
        print("[{0:{1}s}] {2:.1f}%".format('=' * int(i / (self.max / self.len)), self.len, (i / self.max) * 100),
              end='')
        sys.stdout.flush()
