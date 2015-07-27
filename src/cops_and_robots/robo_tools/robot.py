#!/usr/bin/env python
"""Generic definition of a robot.

Currently subclasses from the iRobotCreate class to allow for
physical control of an iRobot Create base (if the Robot class is
configured to control hardware) but could be subclassed to use other
physical hardware in the future.

A robot has a planner that allows it to select goals and a map to
keep track of other robots, feasible regions to which it can move,
an occupancy grid representation of the world, and role-specific
information (such as a probability layer for the rop robot to keep
track of where robber robots may be).

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
import math
import random
import numpy as np

from shapely.geometry import Point

from cops_and_robots.robo_tools.pose import Pose
from cops_and_robots.robo_tools.iRobot_create import iRobotCreate
from cops_and_robots.robo_tools.planner import (MissionPlanner,
                                                GoalPlanner,
                                                PathPlanner,
                                                Controller)
from cops_and_robots.map_tools.map import Map
from cops_and_robots.map_tools.map_elements import MapObject


class Robot(iRobotCreate):
    """Class definition for the generic robot object.

    .. image:: img/classes_Robot.png

    Parameters
    ----------
    name : str
        The robot's name.
    pose : array_like, optional
        The robot's initial [x, y, theta] in [m,m,degrees] (defaults to
        [0, 0.5, 0]).
    map_name : str, optional
        The name of the map (defaults to 'fleming').
    role : {'robber','cop'}, optional
        The robot's role in the cops and robbers game.
    status : two-element list of strings, optional
        The robot's initial mission status and movement status. Cops and
        robbers share possible movement statuses, but their mission statuses
         differ entirely. Defaults to ['on the run', 'without a goal'].
    planner_type: {'simple', 'particle', 'MAP'}, optional
        The robot's type of planner.
    consider_others : bool, optional
        Whether this robot generates other robot models (e.g. the primary cop
        will imagine other robots moving around.) Defaults to false.
    **kwargs
        Arguments passed to the ``MapObject`` attribute.

    """

    def __init__(self,
                 name,
                 pose=None,
                 pose_source='python',
                 color_str='darkorange',
                 map_cfg={},
                 create_mission_planner=True,
                 goal_planner_cfg={},
                 path_planner_cfg={},
                 **kwargs):

        # Object attributes
        self.name = name
        self.pose_source = pose_source

        # Setup map
        self.map = Map(**map_cfg)

        # If pose is not given, randomly place in feasible layer.
        feasible_robot_generated = False
        if pose is None:
            while not feasible_robot_generated:
                x = random.uniform(self.map.bounds[0], self.map.bounds[2])
                y = random.uniform(self.map.bounds[1], self.map.bounds[3])
                if self.map.feasible_layer.pose_region.contains(Point([x, y])):
                    feasible_robot_generated = True
            theta = random.uniform(0, 359)
            pose = [x, y, theta]

        self.pose2D = Pose(self, pose, pose_source)
        self.pose_history = np.array(([0, 0, 0], self.pose2D.pose))
        if pose_source == 'python':
            self.publish_to_ROS = False
        else:
            self.publish_to_ROS = True

        # Setup planners
        if create_mission_planner:
            self.mission_planner = MissionPlanner(self)
        self.goal_planner = GoalPlanner(self,
                                        **goal_planner_cfg)
        # If pose_source is python, this robot is just in simulation
        if not self.publish_to_ROS:
            self.path_planner = PathPlanner(self, **path_planner_cfg)
            self.controller = Controller(self)

        # Define MapObject
        shape_pts = Point([0, 0]).buffer(iRobotCreate.DIAMETER / 2)\
            .exterior.coords
        self.map_obj = MapObject(self.name, shape_pts[:], has_spaces=False,
                                 blocks_camera=False, color_str=color_str)
        self.update_shape()

    def update_shape(self):
        """Update the robot's map_obj.
        """
        self.map_obj.move_absolute(self.pose2D.pose)

    def update(self, i=0):
        """Update all primary functionality of the robot.

        This includes planning and movement for both cops and robbers,
        as well as sensing and map animations for cops.

        Parameters
        ----------
        i : int, optional
            The current animation frame. Default is 0 for non-animated robots.

        Returns
        -------
        tuple or None
            `None` if the robot does not generate an animation packet, or a
            tuple of all animation parameters otherwise.
        """
        if self.pose_source == 'tf':
            self.pose2D.tf_update()

        if self.mission_planner.mission_status is not 'stopped':
            # Update statuses and planners
            self.mission_planner.update()
            self.goal_planner.update()
            if self.publish_to_ROS is False:
                self.path_planner.update()
                self.controller.update()

            # Add to the pose history, update the map
            self.pose_history = np.vstack((self.pose_history,
                                           self.pose2D.pose[:]))
            self.update_shape()

###############################################################################
# Custom Robot classes
###############################################################################


class ImaginaryRobot(object):
    """An imaginary robber for the cop
        Represents what the cop thinks the robber is doing.
        Includes robber's real pose for psuedo detection.
    """
    def __init__(self, name, pose=None):
        self.name = name
        self.pose2D = pose


class Distractor(Robot):
    """The Distractor subclass of the generic robot type.

    Distractors act as distractions during search. They can be given
    move goals, but do not interact with other robots

    Parameters
    ----------
    name : str
        The distractor's name.
    pose : list of float, optional
        The robots's initial [x, y, theta] (defaults to [0, 0.5, 90]).
    planner_type: {'simple', 'particle', 'MAP'}
        The robot's own type of planner.

    Attributes
    ----------
    planner

    """
    mission_planner_defaults = {}
    goal_planner_defaults = {'type_': 'stationary'}
    path_planner_defaults = {'type_': 'direct'}

    def __init__(self,
                 name,
                 pose=[0, 0, 90],
                 pose_source='python',
                 map_cfg={},
                 mission_planner_cfg={},
                 goal_planner_cfg={},
                 path_planner_cfg={},
                 **kwargs):
        # Use class defaults for kwargs not included
        mp_cfg = Distractor.mission_planner_defaults.copy()
        mp_cfg.update(mission_planner_cfg)
        gp_cfg = Distractor.goal_planner_defaults.copy()
        gp_cfg.update(goal_planner_cfg)
        pp_cfg = Distractor.path_planner_defaults.copy()
        pp_cfg.update(path_planner_cfg)

        # Superclass and compositional attributes
        super(Distractor, self).__init__(name,
                                         pose=pose,
                                         pose_source=pose_source,
                                         goal_planner_cfg=gp_cfg,
                                         path_planner_cfg=pp_cfg,
                                         map_cfg=map_cfg,
                                         color_str='darkgreen')
