#!/usr/bin/env python
"""The planner module manages goals and goal paths for the robot.

The planner can be either: stationary, in which it doesn't move;
trajectory, in which it follows a preset path;
simple, in which it picks a random feasible point as a goal pose;
particle-based, in which it selects a goal pose from a particle cloud;
probability-based, in which it selects a goal pose from a
continuous probability distribution;
or specialized, a planner implemented in the owner class (not implemented)

The goal path generated uses the A* planning algorithm, and the
goal pose is generated as the best location from which the robot
can view its target.

"""
__author__ = ["Matthew Aitken", "Nick Sweet"]
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Matthew Aitken", "Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Matthew Aitken"
__email__ = "matthew@raitken.net"
__status__ = "Development"

import logging
import math
import random
from itertools import chain

import numpy as np
from shapely.geometry import Point, LineString

import cops_and_robots.robo_tools.a_star as a_star
from cops_and_robots.map_tools.occupancy_layer import OccupancyLayer


class MissionPlanner(object):
    """The MissionPlanner is responsible for high level planning.


    """
    def __init__(self, robot, mission_status='Moving', target=None):
        self.robot = robot
        self.target = target
        self.trajectory = self.test_trajectory()
        self.mission_status = mission_status

    def stop_all_movement(self):
        self.robot.goal_planner.goal_status = 'done'
        if self.robot.publish_to_ROS:
            self.robot.goal_planner.goal_pose = self.robot.pose2D.pose[:]
            self.robot.goal_planner.create_ROS_goal_message()
        else:
            self.robot.path_planner.planner_status = 'not planning'
            self.robot.controller.controller_status = 'waiting'

        logging.warn('{} has stopped'.format(self.robot.name))
        self.mission_status = 'stopped'

    def test_trajectory(self):
        test_trajectory = iter(np.array([[0, 0], [3.3, 0], [3.3, 2.5],
                                         [3.3, -3], [-6, -3], [-3, -3],
                                         [-3, 0], [-8, 0], [-8, -3],
                                         [-8, 2.5], [-1, 2.5], [-1, 0],
                                         [0, 0]]))
        return test_trajectory

    def update(self):
        """Updates the MissionPlanner.

        This will usually be overwritten when the MissionPlanner is subclassed.

        """
        pass


class GoalPlanner(object):
    """The GoalPlanner class generates goal poses for a robot.

    The goal poses are generated either as the exact target pose, or as a
    view pose from which to see the target.

    Attributes
    ----------
    types : {'simple','trajectory,'particle','MAP'}
        A collection of all methods in which the goal planner finds its goal
        point:
            * a `simple` planner randomly picks a point in the feasible
            region.
            * a 'trajectory' planner uses a predefined trajectory to generate
            goal poses. The trajectory is expected to be provided
            by the mission_planner
            * a `particle` planner selects the particle with greatest
            probability (or randomly selects from the particles that share
            the greatest probability if more than one exists.)
            * a `MAP` or maximum a posteriori planner uses the point of
            greatest posterior probability as the goal point.

    Parameters
    ----------
    type_ : {'simple','trajectory','particle','MAP'}
        The choice of planner type.
    feasible_layer : FeasibleLayer
        A layer object providing both permissible point regions for any object
        and permissible pose regions for any robot with physical dimensions.
    view_distance : float, optional
        The distance in meters from the target goal to place the view goal.
        Default is 0.3m.
    use_target_as_goal : bool, optional
        Use the target location as a goal, rather than using a view pose as
        the goal.

    """
    types = ['stationary', 'simple', 'trajectory', 'particle', 'MAP']
    goal_statuses = ['stuck',
                     'at goal',
                     'moving to goal',
                     'without a goal',
                     'done'
                     ]

    def __init__(self, robot, type_='stationary', view_distance=0.3,
                 use_target_as_goal=True, goal_pose_topic=None):
        if type_ not in GoalPlanner.types:
            raise ValueError('{} is not a type from the list '
                             'of acceptable types: {}'
                             .format(type_, GoalPlanner.types))

        self.robot = robot
        if self.robot.publish_to_ROS:
            import rospy
            from geometry_msgs.msg import PoseStamped
            self.goal_pose_topic = goal_pose_topic
            if self.goal_pose_topic is None:
                self.goal_pose_topic = '/' + self.robot.name.lower() + \
                                       '/move_base_simple/goal'
            self.pub = rospy.Publisher(self.goal_pose_topic, PoseStamped,
                                       queue_size=10)

        self.goal_status = 'without a goal'
        self.type = type_
        self.feasible_layer = robot.map.feasible_layer
        self.view_distance = view_distance
        self.use_target_as_goal = use_target_as_goal
        self.stuck_distance = 0.1  # [m] distance traveled before assumed stuck
        self.stuck_buffer = 200  # time steps after being stuck before checking
        self.stuck_count = self.stuck_buffer
        self.distance_allowance = 0.15  # [m] acceptable distance to a goal
        self.rotation_allowance = 0.5  # [deg] acceptable rotation to a goal

    def find_goal_pose(self):
        """Find a goal pose, agnostic of planner type.

        Parameters
        ----------
        Returns
        -------
        array_like
            A pose as [x,y,theta] in [m,m,degrees].

        """
        if self.type == 'stationary':
            target_pose = None
        elif self.type == 'simple':
            target_pose = self.find_goal_simply()
        elif self.type == 'trajectory':
            target_pose = self.find_goal_from_trajectory()
        elif self.type == 'particle':
            target_pose = self.find_goal_from_particles()
        elif self.type == 'MAP':
            target_pose = self.find_goal_from_probability()

        if target_pose is None:
            return target_pose

        if self.use_target_as_goal:
            goal_pose = target_pose
            logging.info("New goal: {}".format(["{:.2f}".format(a) for a in
                                                goal_pose]))
        else:
            goal_pose = self.view_goal(target_pose)
            logging.info("New view goal ({}): {} to see {}"
                         .format(self.type,
                                 ["{:.2f}".format(a) for a in goal_pose],
                                 ["{:.2f}".format(a) for a in target_pose]))
        return goal_pose

    def view_goal(self, target_pose):
        """Generate a goal as a view pose from which to see the target.

        Translate a target's position to a feasible goal pose for the robot,
        such that the robot can easily see the target. It is not expected
        that the target's heading matters (i.e. the view pose need only
        capture the target, not the target from any side).

        Parameters
        ----------
        target_pose : array_like
            The target pose as [x,y,theta] in [m,m,degrees].

        Returns
        -------
        array_like
            A pose as [x,y,theta] in [m,m,degrees].

        """
        # <>TODO: Extend this to deterministically view the target's face(s)

        # Define a circle as view_distance from the target_pose
        view_circle = Point(target_pose[0:2])\
            .buffer(self.view_distance).exterior

        # Find any feasible point on the view_circle
        feasible_arc = self.feasible_layer\
            .pose_region.intersection(view_circle)
        pt = feasible_arc.representative_point()
        theta = math.atan2(target_pose[1] - pt.y,
                           target_pose[0] - pt.x)  # [rad]
        theta = math.degrees(theta) % 360
        goal_pose = pt.x, pt.y, theta

        return goal_pose

    def find_goal_simply(self):
        """Find a random goal pose on the map.

        Find a random goal pose within map boundaries, residing in the
        feasible pose regions associated with the map.

        Returns
        -------
        array_like
            A pose as [x,y,theta] in [m,m,degrees].

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

        goal_pose = [x, y, theta]
        return goal_pose

    def find_goal_from_particles(self):
        """Find a goal from the most likely particle(s).

        Find a goal pose taken from the particle with the greatest associated
        probability. If multiple particles share the maximum probability, the
        goal pose will be randomly selected from those particles.

        Parameters
        ----------
        fusion_engine : FusionEngine
            A fusion engine with a particle filter.

        Returns
        -------
        array_like
            A pose as [x,y,theta] in [m,m,degrees].

        """
        target = self.robot.mission_planner.target
        fusion_engine = self.robot.fusion_engine
        # <>TODO: @Nick Test this!
        if fusion_engine.filter_type != 'particle':
                raise ValueError('The fusion_engine must have a '
                                 'particle_filter.')

        theta = random.uniform(0, 360)

        # If no target is specified, do default behavior
        if target is None:
            if len(fusion_engine.filters) > 1:
                particles = fusion_engine.filters['combined'].particles
            else:
                particles = next(fusion_engine.filters.iteritems()).particles
        else:
            try:
                particles = fusion_engine.filters[target].particles
                logging.info('Looking for {}'.format(target))
            except:
                logging.warn('No particle filter found for specified target')
                return None

        max_prob = particles[:, 0].max()
        max_particle_i = np.where(particles[:, 0] == max_prob)[0]
        max_particles = particles[max_particle_i, :]

        # Select randomly from max_particles
        max_particle = random.choice(max_particles)
        goal_pose = np.append(max_particle[1:3], theta)

        return goal_pose

    def find_goal_from_probability(self):
        """Find a goal pose from the point of highest probability (the
            Maximum A Posteriori, or MAP, point).

        Parameters
        ----------
        fusion_engine : FusionEngine
            A fusion engine with a probabilistic filter.

        Returns
        -------
        array_like
            A pose as [x,y,theta] in [m,m,degrees].

        """
        target = self.robot.mission_planner.target
        fusion_engine = self.robot.fusion_engine

        # <>TODO: @Nick Test this!
        if fusion_engine.filter_type != 'guass sum':
                raise ValueError('The fusion_engine must have a '
                                 'guass sum filter.')

        theta = random.uniform(0, 360)

        # If no target is specified, do default behavior
        if target is None:
            if len(fusion_engine.filters) > 1:
                posterior = fusion_engine.filters['combined'].probability
            else:
                posterior = next(fusion_engine.filters.iteritems()).probability
        else:
            try:
                posterior = fusion_engine.filters[target].probability
                logging.info('Looking for {}'.format(target))
            except:
                logging.warn('No guass sum filter found for specified target')
                return None

        # <>TODO: softmax update not inside if point in object
        bounds = self.feasible_layer.bounds
        MAP_point, MAP_prob = posterior.max_point_by_grid(bounds)

        # Select randomly from max_particles
        goal_pose = np.append(MAP_point, theta)

        return goal_pose

    def find_goal_from_trajectory(self):
        """Find a goal pose from a set trajectory, defined by the
            mission_planner.
            Trajectory must be a iterated numpy array

        Parameters
        ----------
        Returns
        -------
        array_like
            A pose as [x,y,theta] in [m,m,degrees].

        """
        trajectory = self.robot.mission_planner.trajectory
        try:
            next_goal = next(trajectory)
        except NameError:
            logging.warn('Trajectory not found')
            return None
        except StopIteration:
            logging.info('The specified trajectory has ended')
            return None
        except:
            logging.warn('Unknown Error in loading trajectory')
            return None

        if next_goal.size == 2:
            # if theta is not specified, don't rotate
            current_pose = self.robot.pose2D.pose[0:2]
            theta = math.atan2(next_goal[1] - current_pose[1],
                               next_goal[0] - current_pose[0])  # [rad]
            theta = math.degrees(theta) % 360
            goal_pose = np.append(next_goal, theta)
        else:
            goal_pose = next_goal

        return goal_pose

    def is_stuck(self):
        """Check if the robot has not moved significantly.

        Evaluated over some n time steps. If the robot has not
        existed for n time steps, it is assumed to not be stuck.
        """

        # Check the buffer
        if self.stuck_count > 0:
            self.stuck_count += -1
            logging.debug('Stuck_count = {}'.format(self.stuck_count))
            return False

        self.distance_travelled = 0
        last_poses = self.robot.pose_history[-self.stuck_buffer:]
        for i, pose in enumerate(last_poses):
            if i < self.stuck_buffer - 1:
                dist = math.sqrt((last_poses[i + 1][0] - pose[0]) ** 2 +
                                 (last_poses[i + 1][1] - pose[1]) ** 2)
                self.distance_travelled += dist

        # Restart the counter
        self.stuck_count = self.stuck_buffer
        logging.debug('{} travelled {:.2f}m in last {}'
                      .format(self.robot.name, self.distance_travelled,
                              self.stuck_buffer))
        if self.distance_travelled < self.stuck_distance:
            logging.warn('{} got stuck!'.format(self.robot.name))
            return True
        else:
            return False

    def is_at_goal(self):
        """Check if the robot is near its goal in both distance and rotation.

        Once the robot is near its goal, it rotates towards the final goal
        pose. Once it reaches this goal pose (within some predetermined
        rotation allowance), it is deemed to be at its goal.

        Returns
        -------
        bool
            True if the robot is at its goal, false otherwise.
        """

        goal_pt = Point(self.goal_pose[0:2])
        approximation_circle = goal_pt.buffer(self.distance_allowance)
        pose_pt = Point(self.robot.pose2D.pose[0:2])
        position_bool = approximation_circle.contains(pose_pt)
        logging.debug('Position is {}'.format(position_bool))

        degrees_to_goal = abs(self.robot.pose2D.pose[2] - self.goal_pose[2])
        orientation_bool = (degrees_to_goal < self.rotation_allowance)
        # <>TODO: Replace with ros goal message check
        if self.robot.publish_to_ROS is True:
            orientation_bool = True
        logging.debug('Orientation is {}'.format(orientation_bool))

        logging.debug('is_at_goal = {}'.format((position_bool and
                                                orientation_bool)))
        return position_bool and orientation_bool

    def create_ROS_goal_message(self):
            import tf
            import rospy
            from geometry_msgs.msg import PoseStamped
            self.move_base_goal = PoseStamped()
            theta = np.rad2deg(self.goal_pose[2])
            quaternions = tf.transformations.quaternion_from_euler(0, 0, theta)
            self.move_base_goal.pose.position.x = self.goal_pose[0]
            self.move_base_goal.pose.position.y = self.goal_pose[1]
            self.move_base_goal.pose.orientation.x = quaternions[0]
            self.move_base_goal.pose.orientation.y = quaternions[1]
            self.move_base_goal.pose.orientation.z = quaternions[2]
            self.move_base_goal.pose.orientation.w = quaternions[3]
            self.move_base_goal.header.frame_id = '/map'
            self.move_base_goal.header.stamp = rospy.Time.now()
            self.pub.publish(self.move_base_goal)

    def update(self):
        """Checks to see if goal has been reached, assigns new goals,
            and updates goal status

        The robot can be:
            1. without a goal (in need of a goal pose and path)
            2. moving to goal (translating along its goal path)
            3. stuck (hasn't moved recently)
            4. at goal (reached its goal pose)
            5. done (no longer taking any goals)
        """
        current_status = self.goal_status
        new_status = current_status

        if current_status == 'without a goal':
            self.goal_pose = self.find_goal_pose()
            if self.goal_pose is None:
                new_status = 'done'
            else:
                if self.robot.publish_to_ROS is True:
                    self.create_ROS_goal_message()
                else:
                    self.robot.path_planner.path_planner_status = 'planning'
                new_status = 'moving to goal'

        elif current_status == 'moving to goal':
            if self.is_stuck():
                new_status = 'stuck'
            elif self.is_at_goal():
                new_status = 'at goal'

        elif current_status == 'stuck':
            prev_type = self.type
            self.type = 'simple'
            self.goal_pose = self.find_goal_pose()
            self.type = prev_type
            if self.robot.publish_to_ROS is True:
                self.create_ROS_goal_message()
            else:
                self.robot.path_planner.path_planner_status = 'planning'
            new_status = 'moving to goal'

        elif current_status == 'at goal':
            new_status = 'without a goal'

        if current_status != new_status:
            logging.info("{}'s goal_status changed from {} to {}."
                         .format(self.robot.name, current_status, new_status))

        self.goal_status = new_status


class PathPlanner(object):
    """The PathPlanner class generates a path to a goal.

    The paths are generated as LineStrings (from Shapely) and
    a final orientation, and passed to the Controller class as numpy arrays

    Attributes
    ----------
    types : {'direct', 'a_star'}
        A collection of all methods in which the path planner finds its paths:
            * a `direct` planner creates a straight line path between
            the robot's current pose and the goal pose.
            * a 'a_star' planner creates a trajectory to the goal pose using an
            occupancy grid and the A* path finding algorithm

    Parameters
    ----------
    type_ : {'direct', 'a_star'}
        The choice of planner type.

    """
    types = ['direct', 'a_star']
    path_planner_statuses = ['not planning', 'planning']

    def __init__(self, robot, type_='direct'):
        if type_ not in PathPlanner.types:
            raise ValueError('{} is not a type from the list '
                             'of acceptable types: {}'
                             .format(type_, PathPlanner.types))
        self.type = type_
        self.robot = robot
        self.path_planner_status = 'not planning'

        if self.type == 'a_star':
            self.cell_size = 0.1
            # Generate Occupancy Grid from feasible layer
            self.occupancy_layer = OccupancyLayer(
                cell_size=self.cell_size,
                feasible_layer=self.robot.map.feasible_layer,
                bounds=self.robot.map.feasible_layer.bounds)

    def find_path(self):
        """Find path to a goal_pose, agnostic of planner type.

        Returns
        -------
        array_like
            Returns both a complete goal path (as a Shapely `LineString`) and
            the final pose angle

        """
        if self.type == 'direct':
            goal_path, final_theta = self.find_path_directly()
        elif self.type == 'a_star':
            goal_path, final_theta = self.find_path_from_a_star()

        self.goal_path = goal_path
        self.final_theta = final_theta

    def find_path_directly(self):
        """Finds a path directly to the goal

        """
        current_pose = np.array(self.robot.pose2D.pose[0:2])
        goal_pose = np.array(self.robot.goal_planner.goal_pose[:])
        goal_path = np.array([current_pose, goal_pose])

        final_theta = np.array([goal_pose[2] % 360])  # degrees
        return goal_path, final_theta

    def find_path_from_a_star(self):
        """Finds a path using the A* search algorithm

            a_star takes an occupancy_grid[y][x] where 1 is an obstacle, 0
            is a feasible area. y and x are integer indexes, so the resulting
            points must be translated to corresponding floating positions.

            dirs allows for a choice between moving in 90 degree steps or
            45 degree steps.

        """
        current_point = self.robot.pose2D.pose[0:2]
        goal_point = self.robot.goal_planner.goal_pose[0:2]
        final_theta = np.array([self.robot.goal_planner.goal_pose[2] % 360])

        dirs = 8  # number of possible directions to move on the map
        if dirs == 4:
            dx = [1, 0, -1, 0]
            dy = [0, 1, 0, -1]
        elif dirs == 8:
            dx = [1, 1, 0, -1, -1, -1, 0, 1]
            dy = [0, 1, 1, 1, 0, -1, -1, -1]

        n = len(self.occupancy_layer.x_coords)
        m = len(self.occupancy_layer.y_coords)

        # Return the index of the closest discretized value to current_point
        start = []
        end = []
        start = [np.abs(self.occupancy_layer.x_coords - current_point[0]).argmin(),
                 np.abs(self.occupancy_layer.y_coords - current_point[1]).argmin()]
        end = [np.abs(self.occupancy_layer.x_coords - goal_point[0]).argmin(),
               np.abs(self.occupancy_layer.y_coords - goal_point[1]).argmin()]

        # Find a route (a series of direction choices) using A*
        route = a_star.pathFind(self.occupancy_layer.occupancy_grid, n, m,
                                dirs, dx, dy, start[0], start[1],
                                end[0], end[1])

        if route == 'No Path':
            logging.warn('No path found')
            # <>TODO: Kick it to simple planner without waiting
            # sit still so simple planner kicks in
            goal_path = np.array([current_point])
            return goal_path, final_theta

        # Convert to a list of points
        x = start[0]
        y = start[1]
        path = []
        for i in range(len(route)):
            j = int(route[i])
            x += dx[j]
            y += dy[j]
            path.append([x, y])

        # Convert to a list of coordinates
        path = np.array(path)
        # print self.occupancy_layer.bounds[0:2]
        path = (path * self.occupancy_layer.cell_size
                + np.array(self.occupancy_layer.bounds[0:2]))

        current_point = np.array([current_point])
        goal_point = np.array([goal_point])
        goal_path = np.concatenate((current_point, path, goal_point))

        return goal_path, final_theta

    def update(self):
        current_status = self.path_planner_status
        new_status = current_status

        if current_status == 'planning':
            # Find new path
            self.find_path()
            # Update controller
            self.robot.controller.goal_path = self.goal_path
            self.robot.controller.final_theta = self.final_theta
            self.robot.controller.controller_status = 'updating path'
            # Update Status
            new_status = 'not_planning'

        self.path_planner_status = new_status

        if new_status != current_status:
            logging.debug('{}\'s path planning status changed from {} to {}'
                          .format(self.robot.name, current_status, new_status))


class Controller(object):
    """The Controller class generates moves the robot through a path."""
    def __init__(self, robot):
        self.max_move_distance = 0.2  # [m] per time step
        self.max_rotate_distance = 15  # [deg] per time step
        self.robot = robot
        self.distance_allowance = 0.1  # [m] acceptable distance to a goal
        self.rotation_allowance = 0.5  # [deg] acceptable rotation to a goal
        self.controller_status = 'waiting'  # Start by waiting
        self.goal_path = None
        self.final_theta = None

    def translate_to_pose(self, point):
        """Move the robot's x,y positions towards a goal point.

        Parameters
        ----------
        point: [x, y] array_like

        """
        path = LineString((self.robot.pose2D.pose[0:2], point))
        next_point = path.interpolate(self.max_move_distance)
        self.robot.pose2D.pose[0:2] = (next_point.x, next_point.y)
        logging.debug("{} translated to {}"
                      .format(self.robot.name,
                              ["{:.2f}".format(a) for a
                               in self.robot.pose2D.pose]))

    def rotate_to_pose(self, theta):
        """Rotate the robot about its centroid towards a goal angle.

        Parameters
        ----------
        theta : float
            The goal angle in degrees for the robot to attempt to rotate
            towards. Defaults to the robot's current goal pose angle.

        """
        # Rotate ccw or cw
        angle_diff = theta - self.robot.pose2D.pose[2]
        rotate_ccw = (abs(angle_diff) < 180) and theta > self.robot.pose2D.pose[2] or \
                     (abs(angle_diff) > 180) and theta < self.robot.pose2D.pose[2]
        if rotate_ccw:
            next_angle = min(self.max_rotate_distance, abs(angle_diff))
        else:
            next_angle = -min(self.max_rotate_distance, abs(angle_diff))
        logging.debug('Next angle: {:.2f}'.format(next_angle))
        self.robot.pose2D.pose[2] = (self.robot.pose2D.pose[2] + next_angle) % 360
        logging.debug("{} rotated to {}"
                      .format(self.robot.name,
                              ["{:.2f}".format(a) for a in self.robot.pose2D.pose]))

    def is_near_theta(self):
        """Check if the robot has the correct angle.

        The robot will first rotate to the correct angle before translating
        towards its goal.

        Returns
        -------
        bool
            True if the robot is pointing along the goal path (or final angle),
             false otherwise.
        """
        if self.controller_status != 'rotating':
            raise RuntimeError("*is_on_path* should not be called while the "
                               "robot's status is anything but *rotating*.")

        degree_diff = abs(self.robot.pose2D.pose[2] - self.theta)
        return degree_diff < self.rotation_allowance

    def is_near_waypoint(self):
        """Check if the robot is near its waypoint (in distance, not rotation).

        Returns
        -------
        bool
            True if the robot is near its waypoint, false otherwise.
        """
        if self.controller_status != 'translating':
            raise RuntimeError("*is_near_waypoint* should not be called while "
                               "the robot's status is anything "
                               "but *translating*.")

        way_pt = Point(self.next_waypoint[0:2])
        approximation_circle = way_pt.buffer(self.distance_allowance)
        pose_pt = Point(self.robot.pose2D.pose[0:2])
        return approximation_circle.contains(pose_pt)

    def update_next_waypoint(self):
        """Updates the next waypoint the robot will go to.
            Checks to see if the next_waypoint is the final angle,
            if not, calculates the angle between the current position
            and next_waypoint

        """
        # With no error, self.waypoint = self.next_waypoint
        self.waypoint = self.robot.pose2D.pose[0:2]
        self.next_waypoint = next(self.waypoint_path)
        if self.next_waypoint.size is 1:
            self.theta = self.next_waypoint
        else:
            opposite = self.next_waypoint[1] - self.waypoint[1]
            adjacent = self.next_waypoint[0] - self.waypoint[0]
            self.theta = math.atan2(opposite, adjacent)  # rads
            self.theta = math.degrees(self.theta) % 360  # degs

    def update(self):
        """Defines the robot's current path status and moves appropriately

        The robot can be:
            1. rotating
            2. translating
            3. waiting
            4. updating path

        """
        current_status = self.controller_status
        new_status = current_status

        if self.controller_status == 'updating path':
            # Create a new chain object to iterate through
            self.waypoint_path = chain(self.goal_path, self.final_theta)
            # Set waypoint and next_waypoint
            self.next_waypoint = next(self.waypoint_path)
            self.update_next_waypoint()
            # Start following the path
            new_status = 'rotating'

        elif self.controller_status == 'rotating':
            if self.is_near_theta():
                if self.next_waypoint.size is 1:
                    new_status = 'waiting'
                else:
                    new_status = 'translating'
            else:
                self.rotate_to_pose(self.theta)

        elif current_status == 'translating':
            if self.is_near_waypoint():
                self.update_next_waypoint()
                new_status = 'rotating'
            else:
                self.translate_to_pose(self.next_waypoint)

        self.controller_status = new_status

        if new_status != current_status:
            logging.debug('{}\'s controller status changed from {} to {}'
                          .format(self.robot.name, current_status, new_status))
