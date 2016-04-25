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
                                                     velocity_prior,
                                                     )
from cops_and_robots.fusion.softmax._models import speed_model_2d
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
                 probability_type='grid',
                 velocity_states=False,
                 dynamic_model=True,
                 use_STM=True,
                 ):
        self.target_name = target_name
        self.relevant_targets = ['nothing', 'a robot', self.target_name]
        self.feasible_layer = feasible_layer
        self.motion_model = motion_model
        self.finished = False
        self.dynamic_model = dynamic_model
        self.measurements = []

        # Define the initial prior probability distribution
        if feasible_layer is not None:
            feasible_region = self.feasible_layer.pose_region
        else:
            feasible_region = None

        # Define the probability representation
        if velocity_states == True:
            prior = velocity_prior()
        elif probability_type == 'grid':
            prior = uniform_prior(feasible_region=feasible_region,
                                  use_STM=use_STM)
        elif probability_type == 'particle':
            prior = uniform_particle_prior(feasible_region=feasible_region)
        else:
            prior = fleming_prior(feasible_region=feasible_region)
        self.probability = prior
        self.original_prior = prior

        self.recently_fused_update = False
        self.rosbag_process = rosbag_process

    def update(self, camera=None, human_sensor=None, velocity_state=None):
        if self.finished:
            logging.debug('No need to update - this filter is finished.')
            return

        # Flush I/O for any rosbag process
        try:
            self.rosbag_process.stdin.flush()
            self.rosbag_process.stdout.flush()
        except AttributeError:
            logging.debug('Not playing rosbag')

        # Dynamics Update
        if self.dynamic_model:
            if velocity_state is None:
                self.probability.dynamics_update()
            else:
                self.probability.dynamics_update(velocity_state=velocity_state)

        # Measurement Update
        self._camera_update(camera)
        self._human_update(human_sensor)

    def _camera_update(self, camera):
        if camera is None:
            return
        likelihood = camera.detection_model
        measurement = 'No Detection'

        self.probability.measurement_update(likelihood, measurement, use_LWIS=True)
        self.probability.camera_viewcone = camera.detection_model.poly  # for plotting
        self.recently_fused_update = True

    def _verify_human_update(self, human_sensor):
        """Ensure the update is meaningful and applies to this filter.
        """

        # Stop if the human sensor doesn't have a new statement
        if human_sensor.statement is None:
            return False

        # Stop if the target doesn't apply to this filter
        if human_sensor.statement.target not in self.relevant_targets:
            logging.debug("Measurement about '{}' does not apply to {}."
                          .format(human_sensor.target, self.target_name))
            return False

        return True

    def robber_detected(self, robber_pose):
        """Update the filter for a detected robber.
        """

        # <>TODO: Figure out better strategy when robber detected
        self.probability = GaussianMixture(1, robber_pose[0:2], 0.01 * np.eye(2))
        self.finished = True


class GridFilter(Filter):
    """Grid-based filter
    """
    def __init__(self, *args, **kwargs):
        super(GridFilter, self).__init__(*args, **kwargs)
        
