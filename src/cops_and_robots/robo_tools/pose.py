#!/usr/bin/env python
"""Provides pose ...

"""
__author__ = "Matthew Aitken"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Matthew Aitken", "Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Matthew Aitken"
__email__ = "matthew@raitken.net"
__status__ = "Development"

import logging
import numpy as np


class Pose(object):
    """
    Instance Parameters
    -------------------
    pose_source: {'ROS_TOPIC_NAME', 'python'}
        Specify a ROS odom topic, or python

    Instance Attributes
    -------------------
    pose: list of float
        _pose is used internally, especially with ROS
        pose is to be used externally
    filename_for_recording: str

    Instance Methods
    ----------------
    callback()
    record_pose()

    Global Defaults
    ----------------
    pose_source_default: {'ROS_TOPIC_NAME','python'}

    Class Attributes
    ----------

    Class Methods
    -------------


    If the pose_source is not python, a new node will be created to listen
    to a topic named /pose_source.
    """
    def __init__(self, robot, pose=[0, 0, 0],
                 pose_source='python',
                 filename_for_recording=None):

        self.robot = robot
        self.pose_source = pose_source
        self._pose = pose
        self.filename_for_recording = filename_for_recording

        if pose_source != 'python':
            # Lazy imports
            import rospy
            from nav_msgs.msg import Odometry
            import tf

            if pose_source == 'tf':
                self.listener = tf.TransformListener()
            else:
                rospy.Subscriber(self.pose_source, Odometry, self.callback)

    def callback(self, msg):
        import tf
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        (_, _, theta) = tf.transformations.euler_from_quaternion(
            [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
             msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self._pose = [x, y, np.rad2deg(theta)]
        # logging.info(self.pose)
        # print('Robot Pose')
        # print(self.pose)

    def tf_update(self):
        import rospy
        import tf
        ref = "/" + self.robot.name.lower() + "/odom"
        child = "/" + self.robot.name.lower() + "/base_footprint"
        try:
        	(trans, rot) = self.listener.lookupTransform(ref, child, rospy.Time(0))
        except:
        	logging.error("Can't look up transform")
        	return
        
        x = trans[0]
        y = trans[1]
        (_, _, theta) = tf.transformations.euler_from_quaternion(rot)
        # self._pose = [x, y, np.rad2deg(theta)]
        self._pose = [x, y, np.rad2deg(theta) - 90] #<>TODO: Remove calibration hack!

        # print self._pose

    @property
    def x(self):
        return self._pose[0]

    @x.setter
    def x(self, x):
        self._pose[0] = x
        if self.pose_source != 'python':
            print('You should not set the x during ROS simulation.')

    @property
    def y(self):
        return self._pose[1]

    @y.setter
    def y(self, y):
        self._pose[1] = y
        if self.pose_source != 'python':
            print('You should not set the y during ROS simulation')

    @property
    def theta(self):
        return self._pose[2]

    @theta.setter
    def theta(self, theta):
        self._pose[2] = theta
        if self.pose_source != 'python':
            print('You should not set the theta during ROS simulation')

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, pose):
        self._pose = pose
        if self.pose_source != 'python':
            print('You should not set the pose during ROS simulation')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    deckard = Pose([0, 0, 0], pose_source='odom')

"""
Python use cases:
read pose
update pose

ROS use cases:
read pose
update pose MUST FAIL (look into try/except)
"""
