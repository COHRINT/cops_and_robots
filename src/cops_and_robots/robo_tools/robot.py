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

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``iRobot_create`` for hardware control.
        2. ``planner`` to set goal poses and paths.
        3. ``map`` to maintain environment information.
        4. ``map_obj`` to create its own shape in the map.
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

from cops_and_robots.robo_tools.iRobot_create import iRobotCreate
from cops_and_robots.robo_tools.planner import Planner
from cops_and_robots.map_tools.map import set_up_fleming
from cops_and_robots.map_tools.map_obj import MapObj


class Robot(iRobotCreate):
    """Class definition for the generic robot object.

    Subclassed by either cops or robbers.

    :param name: the robot's name.
    :type name: String.
    :param pose:
    :type pose:
    :param control_hardware:
    :type control_hardware:
    :param default_color:
    :type default_color:
    :param pose:
    :type pose:
    """

    # all_robots = {'Deckard': 'cop',
    #               'Roy': 'robber',
    #               'Leon': 'robber',
    #               'Pris': 'robber',
    #               'Zhora': 'robber',
    #               }

    all_robots = {'Deckard': 'cop',
                  'Roy': 'robber',
                  'Leon': 'robber',
                  }

    movement_statuses = ['stuck',
                         'at goal',
                         'near goal',
                         'on goal path',
                         'rotating',
                         'without a goal'
                         ]

    def __init__(self,
                 name,
                 pose=[0, 0.5, 0],
                 map_name='fleming',
                 role='robber',
                 status=['on the run', 'without a goal'],
                 planner_type='simple',
                 consider_others=False,
                 control_hardware=False,
                 **kwargs):

        # Check robot name
        name = name.capitalize()
        if name not in Robot.all_robots:
            raise ValueError('{} is not a robot name from the list '
                             'of acceptable robots: {}'
                             .format(name, Robot.all_robots))

        # Figure out which robots to consider
        if consider_others:
            self.other_robots = Robot.all_robots
            del self.other_robots[name]
        else:
            self.other_robots = {}

        # Initialize iRobot Create superclass only if controlling hardware
        self.control_hardware = control_hardware
        if self.control_hardware:
            super(Robot, self).__init__()

        # Class attributes
        self.name = name
        self.pose = pose
        self.goal_pose = self.pose[:]
        self.role = role
        self.status = status
        if not map_name:
            self.map = None
        else:
            self.map = set_up_fleming()
        self.planner = Planner(planner_type, self.map.feasible_layer)
        self.fusion_engine = None

        # Movement attributes
        self.move_distance = 0.2  # [m] per time step
        self.rotate_distance = 15  # [deg] per time step
        self.pose_history = np.array(([0, 0, 0], self.pose))
        self.check_last_n = 50  # number of poses to look at before stuck
        self.stuck_distance = 0.1  # [m] distance traveled before assumed stuck
        self.stuck_buffer = 20  # time steps after being stuck before checking
        self.distance_allowance = 0.1  # [m] acceptable distance to a goal
        self.rotation_allowance = 0.5  # [deg] acceptable rotation to a goal
        self.num_goals = None  # number of goals to reach (None for infinite)

        # Define MapObj
        shape_pts = Point(pose[0:2]).buffer(iRobotCreate.DIAMETER / 2)\
            .exterior.coords
        self.map_obj = MapObj(self.name, shape_pts[:], has_zones=False,
                              **kwargs)
        self.update_shape()

        # Add self and others to the map
        if self.role == 'cop':
            self.map.add_cop(self.map_obj)
        else:
            self.map.add_robber(self.map_obj)
        self.make_others()

        # Start with a goal and a path
        self.goal_pose = self.planner.find_goal_pose(self.fusion_engine)
        self.path, self.path_theta = self.planner.update_path(self.pose)

    def update_shape(self):
        """Update the robot's map_obj.
        """

        # <>TODO: refactor this
        self.map_obj.move_shape((self.pose -
                                 self.pose_history[-2:, :]).tolist()[0])

    def make_others(self):
        """Generate robot objects for all other robots.

        Create personal belief (not necessarily true!) of other robots,
        largely regarding their map positions. Their positions are
        known to the 'self' robot, but this function will be expanded
        in the future to include registration between robots: i.e.,
        optional pose and information sharing instead of predetermined
        sharing.
        """

        self.missing_robbers = {}  # all are missing to begin with!
        self.known_cops = {}
        for name, role in self.other_robots.iteritems():

            # Randomly place the other robots
            x = random.uniform(self.map.bounds[0], self.map.bounds[2])
            y = random.uniform(self.map.bounds[1], self.map.bounds[3])
            theta = random.uniform(0, 359)
            pose = [x, y, theta]

            # Add other robots to the map
            if role == 'robber':
                new_robber = robber.Robber(name, pose=pose)
                self.map.add_robber(new_robber.map_obj)
                self.missing_robbers[name] = new_robber
            else:
                new_cop = cop.Cop(name, pose=pose)
                self.map.add_cop(new_cop.map_obj)
                self.known_cops[name] = new_cop

    def translate_towards_goal(self, path=None):
        """Move the robot's x,y positions towards a goal point.

        :param path: a line representing the path to the goal.
        :type path: LineString.
        """

        if not path:
            path = self.path
        next_point = path.interpolate(self.move_distance)
        self.pose[0:2] = (next_point.x, next_point.y)
        logging.debug("{} translated to {}"
                      .format(self.name,
                              ["{:.2f}".format(a) for a in self.pose]))

    def rotate_to_pose(self, theta=None):
        """Rotate the robot about its centroid towards a goal angle.

        Note:
            All angles represented in degrees.
        """
        if not theta:
            theta = self.goal_pose[2]

        # Rotate ccw or cw
        angle_diff = theta - self.pose[2]
        rotate_ccw = (abs(angle_diff) < 180) and theta > self.pose[2] or \
                     (abs(angle_diff) > 180) and theta < self.pose[2]
        if rotate_ccw:
            next_angle = min(self.rotate_distance, abs(angle_diff))
        else:
            next_angle = -min(self.rotate_distance, abs(angle_diff))
        logging.debug('Next angle: {:.2f}'.format(next_angle))
        self.pose[2] = (self.pose[2] + next_angle) % 360
        logging.debug("{} rotated to {}"
                      .format(self.name,
                              ["{:.2f}".format(a) for a in self.pose]))

    def is_stuck(self):
        """Check if the robot has not moved significantly.

        Evaluated overover some n time steps. If the robot has not
        existed for n time steps, it is assumed to not be stuck.
        """

        # Check the buffer
        if self.stuck_buffer > 0:
            self.stuck_buffer += -1
            return False

        if len(self.pose_history) > self.check_last_n:
            distance_travelled = 0
            last_poses = self.pose_history[-self.check_last_n:]
            for i, pose in enumerate(last_poses):
                dist = math.sqrt((last_poses[i][0] - pose[0]) ** 2 +
                                 (last_poses[i][1] - pose[1]) ** 2)
                distance_travelled += dist

            # Update the buffer
            self.stuck_buffer = 20

            logging.debug('{} travelled {:.2f}m in last {}'
                          .format(self.name, distance_travelled,
                                  self.check_last_n))
            return distance_travelled < self.stuck_distance
        else:
            return False

    def has_new_goal(self):
        """Check if the robot received a new goal from the planner.
        """
        # <>TODO: Validate this
        return np.sum((self.pose_history[-1, :], -self.goal_pose[:])) > 0.1
        # return True

    def is_on_path(self):
        """Check if the robot is on the correct path.

        The robot will first rotate to the correct angle before translating
        towards its goal.
        """
        if self.status[1] != 'rotating':
            raise RuntimeError("*is_on_path* should not be called while the "
                               "robot's status is anything but *rotating*.")

        return abs(self.pose[2] - self.path_theta) < self.rotation_allowance

    def is_near_goal(self):
        """Check if the robot is near its goal (in distance, not rotation).

        The robot will only rotate to its final goal pose after it no longer
        needs to translate to be close enough to its goal point. 'Close enough'
        is given by a predetermined distance allowance.
        """
        if self.status[1] != 'on goal path':
            raise RuntimeError("*is_near_goal* should not be called while the "
                               "robot's status is anything but *on goal "
                               "path*.")

        goal_pt = Point(self.goal_pose[0:2])
        approximation_circle = goal_pt.buffer(self.distance_allowance)
        pose_pt = Point(self.pose[0:2])
        return approximation_circle.contains(pose_pt)

    def is_at_goal(self):
        """Check if the robot is near its goal in both distance and rotation.

        Once the robot is near its goal, it rotates towards the final goal
        pose. Once it reaches this goal pose (within some predetermined
        rotation allowance), it is deemed to be at its goal.
        """
        if self.status[1] != 'near goal':
            raise RuntimeError("*is_at_goal* should not be called while the "
                               "robot's status is anything but *near goal*.")

        return abs(self.pose[2] - self.goal_pose[2]) < self.rotation_allowance

    def update_movement_status(self):
        """Define the robot's current movement status.

        The robot can be:
            1. stuck (hasn't moved recently)
            2. at goal (reached its goal pose)
            3. near goal (x and y are close to goal pose, but not theta)
            4. on goal path (translating along its goal path)
            5. rotating (turning to align with path segment)
            6. without a goal (in need of a goal pose and path)
        """

        current_status = self.status[1]
        new_status = current_status

        if current_status == 'stuck':
            new_status = 'without a goal'
        elif current_status == 'without a goal':
            # if self.has_new_goal():
            new_status = 'rotating'
        elif current_status == 'rotating':
            if self.is_on_path():
                new_status = 'on goal path'
        elif current_status == 'on goal path':
            if self.is_near_goal():
                new_status = 'near goal'
        elif current_status == 'near goal':
            if self.is_at_goal():
                new_status = 'at goal'
        elif current_status == 'at goal':
            if True:
                new_status = 'without a goal'

        # Always check if sttuck
        if self.is_stuck() and new_status != 'without a goal':
            new_status = 'stuck'

        if current_status != new_status:
            logging.debug("{}'s status changed from {} to {}."
                          .format(self.name, current_status, new_status))
        self.status[1] = new_status

    def update(self, i=0):
        """Update all primary functionality of the robot.

        This includes planning and movement for both cops and robbers,
        as well as sensing and map animations for cops.
        """

        # If stationary or done, do absolutely nothing.
        if self.status[0] in ('stationary', 'done'):
            return

        # Generate new path and goal poses
        if self.status[1] in ('without a goal'):
            self.goal_pose = self.planner.find_goal_pose(self.fusion_engine)

        # If stuck, find goal simply
        if self.status[1] in ('stuck'):
            prev_type = self.planner.type
            self.planner.type = 'simple'
            self.goal_pose = self.planner.find_goal_pose(self.fusion_engine)
            self.planner.type = prev_type

        # Translate or rotate, depending on status
        if self.status[1] == 'rotating':
            self.rotate_to_pose(self.path_theta)
        elif self.status[1] == 'on goal path':
            self.translate_towards_goal()
        elif self.status[1] == 'near goal':
            self.rotate_to_pose(self.goal_pose[2])

        self.path, self.path_theta = self.planner.update_path(self.pose)

        # Update sensor and fusion information, if a cop
        if self.role == 'cop':
            # Try to visually spot a robber
            for missing_robber in self.missing_robbers.values():
                self.sensors['camera'].detect_robber(missing_robber)

            # Update probability model
            self.fusion_engine.update(self.pose, self.sensors,
                                      self.missing_robbers)

        # Add to the pose history, update the map and status
        self.pose_history = np.vstack((self.pose_history, self.pose[:]))
        self.update_shape()
        self.update_movement_status()
        self.update_mission_status()

        # Export the next animation stream
        if self.role == 'cop' and self.show_animation:
            packet = {}
            if not self.map.combined_only:
                for i, robber_name in enumerate(self.missing_robbers):
                    packet[robber_name] = \
                        self.form_animation_packet(robber_name)
            packet['combined'] = self.form_animation_packet('combined')
            return self.stream.send(packet)

    def form_animation_packet(self, robber_name):
        # Cop-related values
        cop_shape = self.map_obj.shape
        if len(self.pose_history) < self.check_last_n:
            cop_path = np.hsplit(self.pose_history[:, 0:2], 2)
        else:
            cop_path = np.hsplit(self.pose_history[-self.check_last_n:, 0:2],
                                 2)

        camera_shape = self.sensors['camera'].viewcone.shape

        # Robber-related values
        particles = self.fusion_engine.filters[robber_name].particles
        if robber_name == 'combined':
            robber_shape = {name: robot.map_obj.shape for name, robot
                            in self.missing_robbers.iteritems()}
        else:
            robber_shape = self.missing_robbers[robber_name].map_obj.shape

        # Form and return packet to be sent
        packet = (cop_shape, cop_path, camera_shape, robber_shape, particles,)
        return packet

# Import statements left to the bottom because of subclass circular dependency
import cops_and_robots.robo_tools.cop as cop
import cops_and_robots.robo_tools.robber as robber
