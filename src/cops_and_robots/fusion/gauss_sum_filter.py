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
from cops_and_robots.fusion.variational_bayes import VariationalBayes


class GaussSumFilter(object):
    """docstring for GaussSumFilter

    """
    def __init__(self, target_name, feasible_layer=None, motion_model='stationary',
                 v_params=[0, 0.1], state_spec='x y x_dot y_dot'):
        self.target_name = target_name
        self.feasible_layer = feasible_layer  # <>TODO: Do something with this
        self.motion_model = motion_model
        self.finished = False
        self.recieved_human_update = False

        self.probability = fleming_prior()

        # Set up the VB fusion parameters
        self.vb = VariationalBayes()

    def update(self, camera, human_sensor=None):
        if self.finished:
            return
        self.update_mixand_motion()
        self._camera_update(camera)
        self._human_update(human_sensor)

    def _camera_update(self, camera):

        mu, sigma, beta = self.vb.update(measurement='No Detection',
                                         likelihood=camera.detection_model,
                                         prior=self.probability,
                                         use_LWIS=True,
                                         poly=camera.detection_model.poly
                                         )
        gm = GaussianMixture(beta, mu, sigma)
        gm.camera_viewcone = camera.detection_model.poly  # for plotting
        self.probability = gm
        # print 'means \n'
        # print self.probability.means
        # print 'covariance \n'
        # print self.probability.covariances

    def _human_update(self, human_sensor):
        hs = human_sensor
        self.recieved_human_update = False
        if human_sensor.new_update:
            wrong_target = hs.get_measurement(self.target_name)
            if wrong_target:
                return

            if not hasattr(hs.grounding, 'relations') \
                or hs.grounding.name.lower() == 'deckard':
                logging.info("Defining relations because {} didn't have any."
                             .format(hs.grounding.name))
                hs.grounding.define_relations()


            # Position update
            label = hs.relation
            if hs.target_name == 'nothing' and hs.positivity == 'is not':
                label = 'a robot'
            elif hs.target_name == 'nothing' or hs.positivity == 'is not':
                label = 'Not ' + label

            # Perform fusion
            likelihood = hs.grounding.relations.binary_models[hs.relation]
            prior = self.probability
            mu, sigma, beta = hs.vb.update(measurement=label,
                                           likelihood=likelihood,
                                           prior=prior,
                                           use_LWIS=False,
                                           )

            # Weight the posterior by the human's false alarm rate
            gm = GaussianMixture(beta, mu, sigma)
            alpha = hs.false_alarm_prob / 2
            posterior = prior.combine_gms(gm, alpha)


            if posterior is not None:
                self.probability = posterior
                self.recieved_human_update = True


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
        self.recieved_human_update = False

    def update_mixand_motion(self):
        #<>TODO: implement this
        pass