#!/usr/bin/env python
from __future__ import division
"""MODULE_DESCRIPTION"""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import logging
import numpy as np
import scipy as sp
import math
import time

from cops_and_robots.fusion.gaussian_mixture import (GaussianMixture,
                                                     fleming_prior,
                                                     )
from cops_and_robots.fusion.grid import Grid, uniform_prior
from cops_and_robots.fusion.particles import Particles, uniform_particle_prior

class Filter(object):
    """Abstract base class for filter types (particle, gauss-sum, etc.)

    """
    probability_types = ['grid','particle','gaussian_mixture']

    def __init__(self, target_name, feasible_layer=None, 
                 motion_model='stationary',
                 state_spec='x y x_dot y_dot',
                 rosbag_process=None,
                 probability_type='grid'
                 ):
        self.target_name = target_name
        self.relevant_targets = ['nothing', 'a robot', self.target_name]
        self.feasible_layer = feasible_layer
        self.motion_model = motion_model
        self.finished = False
        self.recieved_human_update = False
        self.measurements = []

        # Define the initial prior probability distribution
        feasible_region = self.feasible_layer.pose_region
        if probability_type == 'grid':
            prior = uniform_prior(feasible_region=feasible_region)
        elif probability_type == 'particle':
            prior = uniform_particle_prior(feasible_region=feasible_region)
        else:
            prior = fleming_prior(feasible_region=feasible_region)
        self.probability = prior
        self.original_prior = prior

        self.recently_fused_update = False
        self.rosbag_process = rosbag_process

    def update(self, camera, human_sensor=None, ):
        if self.finished:
            logging.debug('No need to update - this filter is finished.')
            return

        # Flush I/O for any rosbag process
        try:
            self.rosbag_process.stdin.flush()
            self.rosbag_process.stdout.flush()
        except AttributeError:
            logging.debug('Not playing rosbag')

        self.probability.dynamics_update()
        self._camera_update(camera)
        self._human_update(human_sensor)

    def _camera_update(self, camera):
        likelihood = camera.detection_model
        measurement = 'No Detection'

        self.probability.measurement_update(likelihood, measurement)
        self.probability.camera_viewcone = camera.detection_model.poly  # for plotting
        self.recently_fused_update = True

    def _human_update(self, human_sensor):
        """Attempts to perform an update given the human sensor.

        This update will fail if the human does not specifiy the current filter
        as the target or if the measurement is malformed.
        """

        # Validate human sensor statement
        measurement = self._verify_human_update(human_sensor)
        if measurement is None:
            return

        # Halt rosbag
        # <>TODO: have rosbag halt itself
        if self.rosbag_process is not None:
            logging.info('Stopped rosbag to do fusion...')
            self.rosbag_process.stdin.write(' ')  # stop 
            self.rosbag_process.stdin.flush()
            time.sleep(0.5)

        # Parse measurement elements and perform fusion
        measurement_label = measurement['relation']
        relation_class = measurement['relation class']
        grounding = measurement['grounding']
        likelihood = grounding.relations.binary_models[relation_class]

        self.fusion( likelihood, measurement_label, human_sensor)

        # Resume rosbag
        # <>TODO: have rosbag halt itself
        if self.rosbag_process is not None:
            self.rosbag_process.stdin.write(' ')  # start rosbag
            self.rosbag_process.stdin.flush()
            logging.info('Resumed rosbag!')

    def _verify_human_update(self, human_sensor):
        """Ensure the update is properly formed and applies to this filter.
        """
        self.recieved_human_update = False
        if not human_sensor.new_update:
            return

        measurement = human_sensor.get_measurement()

        # Stop if the measurement is malformed
        if measurement is False:
            logging.debug('No measurement to parse.')
            return

        # Stop if the target doesn't apply to this filter
        if measurement['target'] not in self.relevant_targets:
            logging.debug('Target {} is not in {} Looking for {}.'
                .format(measurement['target'], human_sensor.utterance,
                        self.target_name))
            return

        # If the measurement is valid, return it
        return measurement


    def fusion(self, likelihood, measurement, human_sensor):
        # prior = self.probability.copy()
        self.probability.measurement_update(likelihood, measurement)

        #<>TODO: include human false alarm rate
        self.recently_fused_update = True
        self.recieved_human_update = True


    def robber_detected(self, robber_pose):
        """Update the filter for a detected robber.
        """

        # <>TODO: Figure out better strategy when robber detected

        # Find closest particle to target
        # robber_pt = robber_pose[0:2]
        # dist = [math.sqrt((pt[0] - robber_pt[0]) ** 2
        #                   + (pt[1] - robber_pt[1]) ** 2)
        #         for pt in self.particles[:, 1:3]]
        # index = dist.index(min(dist))

        # Set all other particles to 0 probability
        # self.particles[:, 2] = 0
        # self.particles[index] = 1

        self.probability = GaussianMixture(1, robber_pose[0:2], 0.01 * np.eye(2))
        self.finished = True
        self.recieved_human_update = False



class GridFilter(Filter):
    """Grid-based filter
    """
    def __init__(self, *args, **kwargs):
        super(GridFilter, self).__init__(*args, **kwargs)
        
