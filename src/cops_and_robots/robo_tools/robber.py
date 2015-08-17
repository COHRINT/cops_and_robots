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
from cops_and_robots.robo_tools.planner import MissionPlanner


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
    mission_planner_defaults = {}
    goal_planner_defaults = {'type_': 'simple',
                             'use_target_as_goal': True}
    path_planner_defaults = {'type_': 'direct'}

    def __init__(self,
                 name,
                 pose=None,
                 pose_source='python',
                 map_cfg={},
                 mission_planner_cfg={},
                 goal_planner_cfg={},
                 path_planner_cfg={},
                 **kwargs):
        # Use class defaults for kwargs not included
        mp_cfg = Robber.mission_planner_defaults.copy()
        mp_cfg.update(mission_planner_cfg)
        gp_cfg = Robber.goal_planner_defaults.copy()
        gp_cfg.update(goal_planner_cfg)
        pp_cfg = Robber.path_planner_defaults.copy()
        pp_cfg.update(path_planner_cfg)
        super(Robber, self).__init__(name,
                                     pose=pose,
                                     pose_source=pose_source,
                                     goal_planner_cfg=gp_cfg,
                                     path_planner_cfg=pp_cfg,
                                     map_cfg=map_cfg,
                                     create_mission_planner=False,
                                     color_str='red',
                                     **kwargs)

        self.found_robbers = {}
        self.mission_planner = RobberMissionPlanner(self, **mp_cfg)


class RobberMissionPlanner(MissionPlanner):
    """The Cop subclass of the generic MissionPlanner
    """
    mission_statuses = ['on the run', 'captured']

    def __init__(self, robot, mission_status='on the run'):

        super(RobberMissionPlanner, self).__init__(robot,
                                                   mission_status=mission_status)

    def update(self):
        """Update the robber's status

        """
        if self.robot.name in self.robot.found_robbers.keys():
            self.mission_status = 'captured'
        if self.mission_status is 'captured':
            self.stop_all_movement()
