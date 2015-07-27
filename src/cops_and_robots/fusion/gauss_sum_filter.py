#!/usr/bin/env python
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

from cops_and_robots.fusion.gaussian_mixture import (GaussianMixture,
                                                     fleming_prior,
                                                     uniform_prior,
                                                     )

class GaussSumFilter(object):
    """docstring for GaussSumFilter

    """
    def __init__(self, target_name, feasible_layer=None, motion_model='stationary',
                 v_params=[0, 0.1], state_spec='x y x_dot y_dot'):
        self.target_name = target_name
        self.feasible_layer = feasible_layer  #<>TODO: Do something with this
        self.motion_model = motion_model
        self.finished = False

        self.probability = fleming_prior()

    def update(self, camera, human_sensor=None):
        if self.finished:
            return
        self.update_mixand_motion()
        self._camera_update(camera)
        self._human_update(human_sensor)

    def _camera_update(self, camera):
        self.probability = camera.detect('gauss sum', prior=self.probability)


    def _human_update(self, human_sensor):
        if human_sensor.new_update:
            human_sensor.new_update = False
            self.probability = human_sensor.detect(self.target_name, 'gauss sum',
                                                   prior=self.probability)

    def robber_detected(self, robber_pose):
        """Update the particle filter for a detected robber.
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

    def update_mixand_motion(self):
        #<>TODO: implement this
        pass