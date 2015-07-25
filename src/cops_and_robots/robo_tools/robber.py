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
    mission_statuses : {'on the run', 'captured'}
        The possible mission-level statuses of any robber, where:
            * `stationary` means the robber is holding its position;
            * `on the run` means the robber is moving around and avoiding the
                cop;
            * `detected` means the robber knows it has been detected by the
                cop;
            * `captured` means the robber has been captured by the cop and
                is no longer moving.

    """
    mission_statuses = ['on the run', 'captured']
    goal_planner_defaults = {'type_': 'simple',
                             'use_target_as_goal': True}

    def __init__(self,
                 name,
                 pose=None,
                 pose_source='python',
                 map_cfg={},
                 goal_planner_cfg={},
                 **kwargs):
        # Use class defaults for kwargs not included
        gp_cfg = Robber.goal_planner_defaults.copy()
        gp_cfg.update(goal_planner_cfg)
        super(Robber, self).__init__(name,
                                     pose=pose,
                                     pose_source=pose_source,
                                     goal_planner_cfg=gp_cfg,
                                     map_cfg=map_cfg,
                                     mission_status='searching',
                                     color_str='red')

        self.found_robbers = {}
        self.found = False

    def update_mission_status(self):
        # <>TODO: Replace with MissionPlanner
        """Update the robber's status

        """
        if self.name in self.found_robbers.keys():
            self.mission_status = 'captured'
        if self.mission_status is 'captured':
            self.mission_planner.stop_all_movement()


class ImaginaryRobber(object):
    """An imaginary robber for the cop
        Represents what the cop thinks the robber is doing.
        Includes robber's real pose for psuedo detection.
    """
    def __init__(self, name, pose=None):
        self.name = name
        self.pose2D = pose
