#!/usr/bin/env python
"""The planner module manages goals and goal paths for the robot.

The planner can be either: simple, in which it picks a random
feasible point on the map as a goal pose; particle-based, in which
it selects a goal pose from a particle cloud; or probability-based,
in which it selects a goal pose from a continuous probability
distribution.

The goal path generated uses the A* planning algorithm, and the
goal pose is generated as the best location from which the robot
can view its target.

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``feasible_layer`` to generate feasible poses.
        2. ``fusion_engine`` for cop robots to provide complex goals.
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
import random
import math

import numpy as np
from shapely.geometry import Point, LineString


class Planner(object):
    """
    :param type_: Type of planner used.
    :type type_: String.
    :param feasible_layer: a map of feasible regions.
    :type feasible_layer: FeasibleLayer.
    :param view_distance: Distance from the target to view it.
    :type view_distance: float.
    :param use_target_as_goal: Move to the target instead of viewing it.
    :type use_target_as_goal: bool.

    Note:
        ``goal_pose`` and ``target_pose`` are different if the robot
        wants to observe the ``target_pose`` (i.e. through a viewframe).
    """

    types = ['simple', 'particle', 'MAP']

    def __init__(self, type_, feasible_layer, view_distance=0.3,
                 use_target_as_goal=True):
        if type_ not in Planner.types:
            raise ValueError('{} is not a type from the list '
                             'of acceptable types: {}'
                             .format(type_, Planner.types))

        self.type = type_
        self.feasible_layer = feasible_layer
        self.view_distance = view_distance
        self.use_target_as_goal = use_target_as_goal

    def find_goal_pose(self, fusion_engine=None,):
        """Find a goal pose, agnostic of planner type.

        :param fusion_engine: a probabalistic target tracker.
        :type fusion_engine: FusionEngine.
        :returns: goal pose as x,y,theta.
        :rtype: list of floats.
        """

        if self.type == 'simple':
            target_pose = self.find_goal_simply()
        elif self.type == 'particle':
            target_pose = self.find_goal_from_particles(fusion_engine)
        elif self.type == 'MAP':
            target_pose = self.find_goal_from_probability(fusion_engine)

        if self.use_target_as_goal:
            self.goal_pose = target_pose
            logging.info("New goal: {}".format(["{:.2f}".format(a) for a in
                                                self.goal_pose]))
        else:
            self.goal_pose = self.view_goal(target_pose)
            logging.info("New view goal: {} to see {}"
                         .format(["{:.2f}".format(a) for a in target_pose],
                                 ["{:.2f}".format(a) for a in self.goal_pose]))

        return self.goal_pose

    def update_path(self, current_pose, goal_pose=None):
        """Find path to a goal_pose, agnostic of planner type.

        :param fusion_engine: a probabalistic target tracker.
        :type fusion_engine: FusionEngine.
        :returns: a line representing the path to the goal.
        :rtype: LineString.
        """

        if not goal_pose:
            goal_pose = self.goal_pose
        # Currently use single line segments, but expand to A* later

        self.goal_path = LineString((current_pose[0:2], goal_pose[0:2]))
        self.path_theta = math.atan2(goal_pose[1] - current_pose[1],
                                     goal_pose[0] - current_pose[0])  # [rad]
        self.path_theta = math.degrees(self.path_theta) % 360
        return self.goal_path, self.path_theta

    def view_goal(self, target_pose):
        """Translate a target's position to a feasible goal pose for
            the robot, such that the robot can easily see the target.
            It is not expected that the target's heading matters.

        :param target_pose: target pose as x,y,theta.
        :type target_pose: list of floats.
        :returns: goal pose as x,y,theta.
        :rtype: list of floats.
        """

        # Define a circle as view_distance from the target_pose
        view_circle = Point(target_pose[0:2])\
            .buffer(self.view_distance).exterior

        # Find any feasible point on the view_circle
        feasible_arc = self.feasible_layer\
            .pose_region.intersection(view_circle)
        pt = feasible_arc.representative_point()
        theta = math.atan2(pt.y - target_pose[1],
                           pt.x - target_pose[0])  # [rad]
        theta = math.degrees(theta) % 360
        goal_pose = pt.x, pt.y, theta
        return goal_pose

    def find_goal_simply(self):
        """Find a random goal pose on the map.

        Find a random goal pose within map boundaries, residing in
        the feasible pose regions associated with the map.

        :returns: goal pose as x,y,theta.
        :rtype: list of floats.
        """

        theta = random.uniform(0, 360)

        feasible_point_generated = False
        bounds = self.feasible_layer.bounds
        while not feasible_point_generated:
            x = random.uniform(bounds[0], bounds[2])
            y = random.uniform(bounds[1], bounds[3])
            goal_pt = Point(x, y)
            if self.feasible_layer.pose_region.contains(goal_pt):
                feasible_point_generated = True

        return [x, y, theta]

    def find_goal_from_particles(self, fusion_engine):
        """Find a goal from the most likely particle(s).

        Find a goal pose taken from the particle with the greatest
        associated probability. If multiple particles share the
        maximum probability, the goal pose will be randomly
        selected from those particles.

        :param fusion_engine: a probabalistic target tracker.
        :type fusion_engine: FusionEngine.
        :returns: goal pose as x,y,theta.
        :rtype: list of floats.
        """

        if not fusion_engine.filters['combined']:
                raise ValueError('The fusion_engine must have a '
                                 'particle_filter.')

        theta = random.uniform(0, 360)

        particles = fusion_engine.filters['combined'].particles
        max_prob = particles[:, 2].max()
        max_particle_i = np.where(particles[:, 2] == max_prob)
        max_particles = particles[max_particle_i, :]

        # Select randomly from max_particles
        max_particle = random.choice(max_particles[0])
        goal_pose = np.append(max_particle[0:2], theta)
        return goal_pose

    def find_goal_from_probability(self, fusion_engine):
        """Find a goal pose from the point of highest probability (the
            Maximum A Posteriori, or MAP, point).

        :param fusion_engine: a probabalistic target tracker.
        :type fusion_engine: FusionEngine.
        :returns: goal pose as x,y,theta.
        :rtype: list of floats.
        """
        pass
