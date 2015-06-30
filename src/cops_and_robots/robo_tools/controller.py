__author__ = "Jeremy Muesing"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Jeremy Muesing", "Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Jeremy Muesing"
__email__ = "jeremy.muesing@colorado.edu"
__status__ = "Development"


import logging
import math
import random
import robot

import numpy as np
from shapely.geometry import Point, LineString


class Controller(object):
    """fill in later"""
    def __init__(self, robot):
        self.max_move_distance = 0.2  # [m] per time step
        self.max_rotate_distance = 15  # [deg] per time step
        self.robot = robot

    def translate_towards_goal(path=None):
        """Move the robot's x,y positions towards a goal point.

        :param path: a line representing the path to the goal.
        :type path: LineString.

        Parameters
        ----------
        path : LineString, optional
            A movement path represented as a Shapely LineString.Defaults to
            the robot's current path.
        """

        if not path:
            path = self.robot.path
        next_point = path.interpolate(self.robot.max_move_distance)
        self.robot.pose[0:2] = (next_point.x, next_point.y)
        logging.debug("{} translated to {}"
                      .format(self.robot.name,
                              ["{:.2f}".format(a) for a in self.robot.pose]))

    def rotate_to_pose(theta=None):
        """Rotate the robot about its centroid towards a goal angle.

        Parameters
        ----------
        theta : float, optional
            The goal angle in degrees for the robot to attempt to rotate
            towards. Defaults to the robot's current goal pose angle.

        """
        if not theta:
            theta = self.robot.next_pose[2]

        # Rotate ccw or cw
        angle_diff = theta - self.robot.pose[2]
        rotate_ccw = (abs(angle_diff) < 180) and theta > self.robot.pose[2] or \
                     (abs(angle_diff) > 180) and theta < self.robot.pose[2]
        if rotate_ccw:
            next_angle = min(self.robot.max_rotate_distance, abs(angle_diff))
        else:
            next_angle = -min(self.robot.max_rotate_distance, abs(angle_diff))
        logging.debug('Next angle: {:.2f}'.format(next_angle))
        self.robot.pose[2] = (self.robot.pose[2] + next_angle) % 360
        logging.debug("{} rotated to {}"
                      .format(self.robot.name,
                              ["{:.2f}".format(a) for a in self.robot.pose]))

        def update (robot):
            if robot.status[1] == 'rotating':
                robot.rotate_to_pose(robot.path_theta)
            elif robot.status[1] == 'on goal path':
                robot.translate_towards_goal()
            elif robot.status[1] == 'near goal':
                robot.rotate_to_pose(robot.goal_pose[2])