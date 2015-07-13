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

from cops_and_robots.fusion.gaussian_mixture import GaussianMixture

class GaussSumFilter(object):
    """docstring for GaussSumFilter

    """
    def __init__(self, target_name, feasible_layer=None, motion_model='stationary',
                 v_params=[0, 0.1], state_spec='x y x_dot y_dot'):
        self.target_name = target_name
        self.feasible_layer = feasible_layer  #<>TODO: Do something with this
        self.motion_model = motion_model
        self.finished = False

        self.probability = GaussianMixture(weights=[32, 14, 15, 14, 14, 14],
                                           means=[[-5.5, 2],  # Kitchen
                                                  [2, 2],  # Billiard Room
                                                  [-4, -0.5],  # Hallway
                                                  [-9, -2.5],  # Dining Room
                                                  [-4, -2.5],  # Study
                                                  [1.5, -2.5],  # Library
                                                  ],
                                           covariances=[[[5.0, 0.0],  # Kitchen
                                                         [0.0, 2.0]
                                                         ],
                                                        [[1.0, 0.0],  # Billiard Rooom
                                                         [0.0, 2.0]
                                                         ],
                                                        [[7.5, 0.0],  # Hallway
                                                         [0.0, 0.5]
                                                         ],
                                                        [[2.0, 0.0],  # Dining Room
                                                         [0.0, 1.0]
                                                         ],
                                                        [[2.0, 0.0],  # Study
                                                         [0.0, 1.0]
                                                         ],
                                                        [[2.0, 0.0],  # Library
                                                         [0.0, 1.0]
                                                         ],
                                                        ])

    def update(self, camera, human_sensor=None):
        if self.finished:
            return
        self.update_mixand_motion()
        self._camera_update(camera)
        # self._human_update(human_sensor)

    def _camera_update(self, camera):
        self.probability = camera.detect('gauss sum', prior=self.probability)

    def _human_update(self, human_sensor):
        self.probability = human_sensor.detect(self.target_name, 'gauss sum',
                                               prior=self.probability)

    def update_mixand_motion(self):
        #<>TODO: implement this
        pass