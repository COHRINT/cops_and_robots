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
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely import affinity

from cops_and_robots.fusion.sensor import Sensor
from cops_and_robots.fusion.softmax import camera_model_2D
from cops_and_robots.fusion.variational_bayes import VariationalBayes
from cops_and_robots.fusion.gaussian_mixture import GaussianMixture

# <>TODO: Remove test stub
from cops_and_robots.map_tools.map_elements import MapObject
from cops_and_robots.map_tools.shape_layer import ShapeLayer


class Camera(Sensor):
    """A conic sensor mounted on a robot.

    The camera provides a viewcone from the point of view of the robot which
    rescales based on its environment (e.g. if it's in front of a wall).

    .. image:: img/classes_Camera.png

    Parameters
    ----------
    robot_pose : array_like, optional
        The cop's initial [x, y, theta] (defaults to [0, 0, 0]).
    visible : bool, optional
        Whether or not the view cone is shown. Default is True.
    default_color : cnames
        Default color to display all camera sensors as. Defaults to yellow.

    """
    def __init__(self, robot_pose=(0, 0, 0), visible=True,
                 default_color=cnames['yellow'], blocking_shapes=Polygon()):
        # Define blocking shapes (updated in shapelayer,
        #   if update_blocking_shapes is true)
        self.blocking_shapes = blocking_shapes
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

        # Create SoftMax model for camera
        self.detection_model = camera_model_2D(self.min_view_dist,
                                               self.max_view_dist)

        # Instantiate Sensor superclass object
        update_rate = 1  # [hz]
        has_physical_dimensions = True
        super(Camera, self).__init__(update_rate, has_physical_dimensions)

        # Set the ideal and actual viewcones
        self.ideal_viewcone = MapObject('Ideal viewcone',
                                        viewcone_pts,
                                        visible=False,
                                        blocks_camera=False,
                                        color_str='pink',
                                        pose=robot_pose,
                                        has_spaces=False,
                                        centroid_at_origin=False,
                                        )
        self.viewcone = MapObject('Viewcone',
                                  viewcone_pts,
                                  visible=True,
                                  blocks_camera=False,
                                  color_str='lightyellow',
                                  pose=robot_pose,
                                  has_spaces=False,
                                  centroid_at_origin=False,
                                  )
        self.view_pose = (0, 0, 0)

        # <>TODO: Add in and test an offset of (-0.1,-0.1)
        self.offset = (0, 0, 0)  # [m] offset (x,y,theta) from center of robot
        self._move_viewcone(robot_pose)

        # Set up the VB fusion parameters
        self.vb = VariationalBayes()

    def update_viewcone(self, robot_pose, shape_layer):
        """Update the camera's viewcone position and scale.

        Parameters
        ----------
        robot_pose : array_like, optional
            The robot's currentl [x, y, theta].
        shape_layer : ShapeLayer
            A layer object providing all the shapes in the map for the camera
            to rescale its viewcone.
        """
        self.blocking_shapes = shape_layer.blocking_shapes
        self._move_viewcone(robot_pose)
        self._rescale_viewcone(robot_pose)

    def _move_viewcone(self, robot_pose):
        """Move the viewcone based on the robot's pose

        Parameters
        ----------
        robot_pose : array_like, optional
            The robot's currentl [x, y, theta].
        """
        pose = (robot_pose[0] + self.offset[0],
                robot_pose[1] + self.offset[1],
                robot_pose[2]
                )

        # Reset the view shape
        self.viewcone.shape = self.ideal_viewcone.shape
        transform = tuple(np.subtract(pose, self.view_pose))
        self.ideal_viewcone.move_relative(transform,
                                          rotation_pt=self.view_pose[0:2])
        self.viewcone.move_relative(transform, rotation_pt=self.view_pose[0:2])
        self.view_pose = pose

    def _rescale_viewcone(self, robot_pose):
        """Rescale the viewcone based on intersecting map objects.

        Parameters
        ----------
        robot_pose : array_like, optional
            The robot's currentl [x, y, theta].
        shape_layer : ShapeLayer
            A layer object providing all the shapes in the map for the camera
            to rescale its viewcone.
        """
        if self.viewcone.shape.intersects(self.blocking_shapes):
            # <>TODO: Use shadows instead of rescaling viewcone
            # calculate shadows for all shapes touching viewcone
            # origin = self.viewcone.project(map_object.shape)
            # shadow = affinity.scale(...) #map portion invisible to the view
            # self.viewcone = self.viewcone.difference(shadow)

            distance = Point(self.view_pose[0:2]).distance(self.blocking_shapes)
            scale = distance / self.max_view_dist * 1.3  # <>TODO: why the 1.3?
            self.viewcone.shape = affinity.scale(self.ideal_viewcone.shape,
                                                 xfact=scale,
                                                 yfact=scale,
                                                 origin=self.view_pose[0:2])
        else:
            self.viewcone.shape = self.ideal_viewcone.shape

    def detect(self, filter_type, particles=None, prior=None):
        """Update a fusion engine's probability from camera detections.

        Parameters
        ----------
        filter_type : {'particle','gauss sum'}
            The type of filter to update.
        particles : array_like, optional
            The particle list, assuming [x,y,p], where x and y are position
            data and p is the particle's associated probability.

        """
        if filter_type == 'particle':
            self._detect_particles(particles)
        else:
            logging.info(prior.means)
            posterior = self._detect_probability(prior)
            logging.info(posterior.means)
            return posterior

    def _detect_particles(self, particles):
        """Update particles based on sensor model.

        particles : array_like
            The particle list, assuming [x,y,p], where x and y are position
            data and p is the particle's associated probability.
        """
        # Translate detection model
        # <>TODO: check this
        self.detection_model.move(self.view_pose)

        # Update particle probabilities in view cone frame
        for i, particle in enumerate(particles):
            if self.viewcone.shape.contains(Point(particle[1:3])):
                particles[i, 0] *= (1 - self.detection_model
                    .probability(state=particle[1:3], class_='Detection'))

        # Renormalize
        particles[:, 0] /= sum(particles[:, 0])

    def _detect_probability(self, prior):

        # Translate detection model
        self.detection_model.move(self.view_pose)

        mu, sigma, beta = self.vb.update(measurement='No Detection',
                                         likelihood=self.detection_model,
                                         prior=prior,
                                         use_LWIS=False
                                         )
        return GaussianMixture(beta, mu, sigma)


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
    wall1 = MapObject('wall1', (l, w), pose=pose)
    pose = (2, 2.2, 0)
    wall2 = MapObject('wall2', (l, w), pose=pose)

    shape_layer = ShapeLayer(bounds=bounds)
    shape_layer.add_obj(wall1)
    shape_layer.add_obj(wall2)
    shape_layer.plot()

    # Define Particle Filter
    # target_pose = (10,10,0)
    # particle_filter = ParticleFilter(bounds=bounds,"Roy")
    # particle_filter.update_viewcone(kinect,target_pose)

    # Move camera and update the camera
    for point in goal_points:
        # kinect.update_viewcone(point,shape_layer,particle_filter,target_pose)
        kinect.update_viewcone(point, shape_layer)
        kinect.viewcone.plot(color=cnames['yellow'],
                             alpha=0.5
                             )

    lim = 4
    ax.set_xlim([-lim / 2, lim])
    ax.set_ylim([-lim / 2, lim])

    # particle_filter.plot()

    plt.show()
