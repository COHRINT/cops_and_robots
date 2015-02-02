#!/usr/bin/env python
"""Provides an visual cone sensor to update the fusion engine.

The camera (mounted at an offset to the cop robot's centroid) is a
cone with an associated probabilty of detection (not necessarily
uniform) provided by its ``sensor`` superclass. The vision cone
reshapes itself at walls (as it cannot see through walls!).

Note:
    Only cop robots have cameras (for now). Robbers may get hardware
    upgreades in future versions, in which case this would be owned by
    the ``robot`` module instead of the ``cop`` module.

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``sensor`` for generic sensor parameters and functions,
           such as update rate and detection chance.
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
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import cnames
from shapely.geometry import Point
from shapely import affinity

from cops_and_robots.robo_tools.fusion.sensor import Sensor

# <>TODO: Remove test stub
from cops_and_robots.map_tools.map_obj import MapObj
from cops_and_robots.map_tools.shape_layer import ShapeLayer


class Camera(Sensor):
    """docstring for Camera"""
    def __init__(self, robot_pose=(0, 0, 0), visible=True,
                 default_color=cnames['yellow']):
        # Define nominal viewcone
        self.min_view_dist = 0.3  # [m]
        self.max_view_dist = 1.0  # [m]
        self.view_angle = math.pi / 2  # [rad]
        viewcone_pts = [(0, 0),
                        (self.max_view_dist * math.cos(self.view_angle / 2),
                         self.max_view_dist * math.sin(self.view_angle / 2)),
                        (self.max_view_dist * math.cos(-self.view_angle / 2),
                         self.max_view_dist * math.sin(-self.view_angle / 2)),
                        (0, 0),
                        ]

        # Instantiate Sensor superclass object
        update_rate = 1  # [hz]
        detection_chance = 0.8
        has_physical_dimensions = True
        super(Camera, self).__init__(update_rate, has_physical_dimensions,
                                     detection_chance)

        # Set the ideal and actual viewcones
        self.ideal_viewcone = MapObj('Ideal viewcone',
                                     viewcone_pts,
                                     visible=False,
                                     default_color_str='pink',
                                     pose=robot_pose,
                                     has_zones=False,
                                     centroid_at_origin=False,
                                     )
        self.viewcone = MapObj('Viewcone',
                               viewcone_pts,
                               visible=True,
                               default_color_str='lightyellow',
                               pose=robot_pose,
                               has_zones=False,
                               centroid_at_origin=False,
                               )
        self.view_pose = (0, 0, 0)
        # <>TODO: Add in and test an offset of (-0.1,-0.1)
        self.offset = (0, 0, 0)  # [m] offset (x,y,theta) from center of robot
        self.move_viewcone(robot_pose)

    def update(self, robot_pose, shape_layer):
        """Update the camera's viewcone based on the robot's position
            in the map.

        :param robot_pose:
        :type robot_pose:
        :param shape_layer:
        :type shape_layer:
        """
        self.move_viewcone(robot_pose)
        self.rescale_viewcone(robot_pose, shape_layer)

    def move_viewcone(self, robot_pose):
        """Move the viewcone based on the robot's pose"""
        pose = (robot_pose[0] + self.offset[0],
                robot_pose[1] + self.offset[1],
                robot_pose[2]
                )

        # Reset the view shape
        self.viewcone.shape = self.ideal_viewcone.shape
        transform = tuple(np.subtract(pose, self.view_pose))
        self.ideal_viewcone.move_shape(transform,
                                       rotation_pt=self.view_pose[0:2])
        self.viewcone.move_shape(transform, rotation_pt=self.view_pose[0:2])
        self.view_pose = pose

    def rescale_viewcone(self, robot_pose, shape_layer):
        all_shapes = shape_layer.all_shapes.buffer(0)  # bit of a hack!
        if self.viewcone.shape.intersects(all_shapes):

            # <>TODO: Use shadows instead of rescaling viewcone
            # calculate shadows for all shapes touching viewcone
            # origin = self.viewcone.project(map_object.shape)
            # shadow = affinity.scale(...) #map portion invisible to the view
            # self.viewcone = self.viewcone.difference(shadow)

            distance = Point(self.view_pose[0:2]).distance(all_shapes)
            scale = distance / self.max_view_dist * 1.3  # <>TODO: why the 1.3?
            self.viewcone.shape = affinity.scale(self.ideal_viewcone.shape,
                                                 xfact=scale,
                                                 yfact=scale,
                                                 origin=self.view_pose[0:2])
        else:
            self.viewcone.shape = self.ideal_viewcone.shape

    def detect(self, fusion_engine_type, particles):
        if fusion_engine_type == 'discrete':
            self.detect_particles(particles)
        else:
            self.detect_probability()

    def detect_robber(self, robber):
        if self.viewcone.shape.contains(Point(robber.pose)):
            robber.status = 'detected'

    def detect_particles(self, particles):
        """ Update particles based on sensor model.
        """
        # Update particle probabilities in view cone
        for i, particle in enumerate(particles):
            if self.viewcone.shape.contains(Point(particle[0:2])):
                particles[i, 2] *= (1 - self.detection_chance)

        # Renormalize
        particles[:, 2] /= sum(particles[:, 2])

    def detect_probability(self):
        pass


if __name__ == '__main__':

    # Pre-test config
    fig = plt.figure(1, figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Define Camera
    kinect = Camera()
    goal_points = [(0, 0, 0),
                   (2, 0, 0),
                   (2, 0.5, 90),
                   (2, 1.5, 90),
                   (2, 1.7, 90),
                   ]

    # Define Map and its objects
    bounds = (-5, -5, 5, 5)

    l = 1.2192  # [m] wall length
    w = 0.1524  # [m] wall width

    pose = (2.4, 0, 90)
    wall1 = MapObj('wall1', (l, w), pose)
    pose = (2, 2.2, 0)
    wall2 = MapObj('wall2', (l, w), pose)

    shape_layer = ShapeLayer(bounds=bounds)
    shape_layer.add_obj(wall1)
    shape_layer.add_obj(wall2)
    shape_layer.plot(plot_zones=False)

    # Define Particle Filter
    # target_pose = (10,10,0)
    # particle_filter = ParticleFilter(bounds=bounds,"Roy")
    # particle_filter.update(kinect,target_pose)

    # Move camera and update the camera
    for point in goal_points:
        # kinect.update(point,shape_layer,particle_filter,target_pose)
        kinect.update(point, shape_layer)
        kinect.viewcone.plot(plot_zones=False,
                             color=cnames['yellow'],
                             alpha=0.5
                             )

    lim = 4
    ax.set_xlim([-lim / 2, lim])
    ax.set_ylim([-lim / 2, lim])

    # particle_filter.plot()

    plt.show()
