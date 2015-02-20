#!/usr/bin/env python
"""Provides some common functionality for robber robots.

While a robber's functionality is largely defined by the ``robot``
module, this module is meant to update the robber's status (which
differs from cop statuses) and provide a basis for any additional
functionality the robbers may need (such as communication between
robbers).

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

from cops_and_robots.robo_tools.robot import Robot


class Robber(Robot):
    """The robber subclass of the generic Robot type.

    .. image:: img/classes_Robber.png

    Parameters
    ----------
    name : str
        The robber's name.
    **kwargs
        Arguments passed to the ``Robber`` superclass.

    Attributes
    ----------
    found : bool
        Whether or not the robber knows its been found.

    Attributes
    ----------
    mission_statuses : {'stationary', 'on the run', 'detected', 'captured'}
        The possible mission-level statuses of any robber, where:
            * `stationary` means the robber is holding its position;
            * `on the run` means the robber is moving around and avoiding the
                cop;
            * `detected` means the robber knows it has been detected by the
                cop;
            * `captured` means the robber has been captured by the cop and
                is no longer moving.

    """
    mission_statuses = ['stationary', 'on the run', 'detected', 'captured']

    def __init__(self, name, **kwargs):
        super(Robber, self).__init__(name=name,
                                     role='robber',
                                     planner_type='simple',
                                     color_str='darkorange',
                                     **kwargs)

        self.found = False

        # <>TODO: break out each robber as its own thread and have them move
        # while not self.found:
        #     self.update()

    def update_status(self):
        """Update the robber's status

        """
        immobile_statuses = ('stationary', 'captured')
        if self.status not in immobile_statuses:
            if self.status == 'detected' and self.time_since_detected < 50:
                self.time_since_detected += 1
            else:
                self.time_since_detected = 0
                self.status = 'on the run'
