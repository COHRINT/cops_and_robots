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
from cops_and_robots.fusion.softmax import (geometric_model,
                                            neighbourhood_model,
                                            product_model, 
                                            )


class GaussSumFilter(object):
    """docstring for GaussSumFilter

    """

    fusion_methods = ['recursive', 'full batch', 'windowed batch', 'grid']
    synthesis_techniques = ['product','geometric', 'neighbourhood']


    def __init__(self, target_name, feasible_layer=None, 
                 motion_model='stationary',
                 v_params=[0, 0.1], 
                 state_spec='x y x_dot y_dot',
                 fusion_method='full batch',
                 synthesis_technique='product',
                 window=2,
                 ):
        self.target_name = target_name
        self.relevant_targets = ['nothing', 'a robot', self.target_name]
        self.feasible_layer = feasible_layer  # <>TODO: Do something with this
        self.motion_model = motion_model
        self.finished = False
        self.recieved_human_update = False
        self.fusion_method = fusion_method
        self.synthesis_technique = synthesis_technique
        self.window = window
        self.measurements = []

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

    def _human_update(self, human_sensor):
        hs = human_sensor
        self.recieved_human_update = False
        if human_sensor.new_update:
            measurement = hs.get_measurement()

            # Break if the target doesn't apply to this filter
            if measurement is False:
                logging.debug('No measurement to parse.')
                return 

            if measurement['target'] not in self.relevant_targets:
                logging.debug('Target {} is not in {} Looking for {}.'
                    .format(measurement['target'], hs.utterance, self.target_name))
                return

            if self.fusion_method == 'recursive':
                self.recursive_fusion(measurement, human_sensor)
            elif self.fusion_method == 'full batch' or 'windowed batch':
                self.batch_fusion(measurement, human_sensor)
            elif self.fusion_method == 'grid':
                self.grid_fusion(measurement, human_sensor)


    def recursive_fusion(self, measurement, human_sensor):
        """Performs fusion once per pass."""
        
        measurement_label = measurement['relation']
        relation_class = measurement['relation class']
        grounding = measurement['grounding']
        likelihood = grounding.relations.binary_models[relation_class]

        self.gm_fusion(likelihood, measurement_label, human_sensor)

    def batch_fusion(self, measurement, human_sensor):
        self.measurements.append(measurement)

        if len(self.measurements) >= self.window:
            
            # Create combined measurement labels
            measurement_labels = []
            for measurement in self.measurements:
                measurement_labels.append(measurement['relation'])
            measurement_label = " + ".join(measurement_labels)

            # Create combined softmax model
            models = []
            for measurement in self.measurements:
                grounding = measurement['grounding']
                relation_class = measurement['relation class']
                model = grounding.relations.binary_models[relation_class]
                models.append(model)

            # Synthesize the likelihood
            if self.synthesis_technique == 'product':
                likelihood = product_model(models)
            elif self.synthesis_technique == 'neighbourhood':
                likelihood = neighbourhood_model(models, measurement_labels)
            elif self.synthesis_technique == 'geometric':
                likelihood = geometric_model(models, measurement_labels)

            # Discard measurements for windowed, increase window size for full
            if self.fusion_method == 'windowed batch':
                self.measurements = []
            elif self.fusion_method == 'full batch':
                self.window += self.window
            self.gm_fusion(likelihood, measurement_label, human_sensor)

    def grid_fusion(self, measurement, human_sensor):
        pass

    def gm_fusion(self, likelihood, measurement_label, human_sensor):
        prior = self.probability
        mu, sigma, beta = self.vb.update(measurement=measurement_label,
                                         likelihood=likelihood,
                                         prior=prior,
                                         use_LWIS=False,
                                         )

        # Weight the posterior by the human's false alarm rate
        gm = GaussianMixture(beta, mu, sigma)
        alpha = human_sensor.false_alarm_prob / 2
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