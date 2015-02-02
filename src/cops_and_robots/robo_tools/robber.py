#!/usr/bin/env python
"""Provides some common functionality for robber robots.

While a robber's functionality is largely defined by the ``robot``
module, this module is meant to update the robber's status (which
differs from cop statuses) and provide a basis for any additional
functionality the robbers may need (such as communication between
robbers).

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``robot`` for all basic functionality.
"""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import logging

from matplotlib.colors import cnames

from cops_and_robots.robo_tools.robot import Robot


class Robber(Robot):
    """The robber subclass of the generic Robot type.

    :param name: the robber's name.
    :type name: String.
    """
    def __init__(self, name, **kwargs):
        super(Robber, self).__init__(name=name,
                                     role='robber',
                                     planner='simple',
                                     default_color=cnames['darkorange'],
                                     **kwargs)

        self.found = False

        # while not self.found:
        #     self.update()

    def update_status(self):
        """Update the robber's status from one of:
            1. stationary
            2. on the run
            3. detected
            4. captured
        """
        immobile_statuses = ('stationary', 'captured')
        if self.status not in immobile_statuses:
            if self.status == 'detected' and self.time_since_detected < 50:
                self.time_since_detected += 1
            else:
                self.time_since_detected = 0
                self.status = 'on the run'
