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
                 fusion_method='recursive',
                 synthesis_technique='geometric',
                 window=1,
                 rosbag_process=None,
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
        self.original_prior = fleming_prior()

        # Set up the VB fusion parameters
        self.vb = VariationalBayes()

        self.recently_fused_update = False
        self.rosbag_process = rosbag_process
        self.human_measurement_count = 0
        self.frame = 0

    def update(self, camera, human_sensor=None, save_file=None):
        if self.finished:
            return

        try:
            self.rosbag_process.stdin.flush()
            self.rosbag_process.stdout.flush()
        except AttributeError:
            logging.debug('Not playing rosbag')

        self.update_mixand_motion()
        self._camera_update(camera)
        self._human_update(human_sensor)
        if save_file is not None:
            if self.recently_fused_update:
                self.save_probability(save_file)
                self.recently_fused_update = False
            self.save_MAP(save_file)

    def save_probability(self, save_file):
        if self.fusion_method == 'windowed batch':
            save_file = save_file + 'windowed_batch_' + str(self.window)
        else:
            save_file = save_file + self.fusion_method.replace(' ', '_')
        filename = save_file + '_' + self.target_name  +'_posterior_' + str(self.human_measurement_count)
        np.save(filename, self.probability)

    def save_MAP(self, save_file):
        self.frame += 1
        bounds = self.feasible_layer.bounds
        try:
            MAP_point, MAP_prob = self.probability.max_point_by_grid(bounds)
        except AttributeError:
            pt = np.unravel_index(self.probability.argmax(), self.X.shape)
            MAP_point = (self.X[pt[0],0],
                         self.Y[0,pt[1]]
                         )
        MAP_point = np.array(MAP_point)


        if self.fusion_method == 'windowed batch':
            save_file = save_file + 'windowed_batch_' + str(self.window)
        else:
            save_file = save_file + self.fusion_method.replace(' ', '_')
        filename = save_file + '_' + self.target_name  +'_MAP_' + str(self.frame)
        np.save(filename, MAP_point)

    def _camera_update(self, camera):
        if self.fusion_method == 'grid':
            self._camera_grid_update(camera)
            return

        mu, sigma, beta = self.vb.update(measurement='No Detection',
                                         likelihood=camera.detection_model,
                                         prior=self.probability,
                                         use_LWIS=True,
                                         poly=camera.detection_model.poly
                                         )
        gm = GaussianMixture(beta, mu, sigma)
        gm.camera_viewcone = camera.detection_model.poly  # for plotting
        self.probability = gm

    def _camera_grid_update(self, camera):
        if not hasattr(self, 'pos'):
            self._set_up_grid()

        if type(self.probability) is GaussianMixture:
            self.probability = self.probability.pdf(self.pos)

        prior_prob = self.probability
        measurement = 'No Detection'
        likelihood_prob = camera.detection_model.probability(state=self.pos,
                                                             class_=measurement
                                                             )

        posterior = likelihood_prob * prior_prob
        posterior /= posterior.sum()

        self.recently_fused_update = True

        self.probability = posterior

    def _human_update(self, human_sensor):
        hs = human_sensor
        self.recieved_human_update = False
        if human_sensor.new_update:

            self.human_measurement_count += 1

            measurement = hs.get_measurement()

            # Break if the target doesn't apply to this filter
            if measurement is False:
                logging.debug('No measurement to parse.')
                return 

            if measurement['target'] not in self.relevant_targets:
                logging.debug('Target {} is not in {} Looking for {}.'
                    .format(measurement['target'], hs.utterance, self.target_name))
                return

            if self.rosbag_process is not None:
                logging.info('Stopped rosbag to do fusion...')
                self.rosbag_process.stdin.write(' ')  # stop 
                self.rosbag_process.stdin.flush()
                import time
                time.sleep(0.5)

            logging.info(self.fusion_method)
            if self.fusion_method == 'recursive':
                self.recursive_fusion(measurement, human_sensor)
            elif self.fusion_method == 'full batch' \
                or self.fusion_method == 'windowed batch':
                self.batch_fusion(measurement, human_sensor)
            elif self.fusion_method == 'grid':
                self.grid_fusion(measurement, human_sensor)

            if self.rosbag_process is not None:
                self.rosbag_process.stdin.write(' ')  # start rosbag
                self.rosbag_process.stdin.flush()
                logging.info('Restarted rosbag!')

    def recursive_fusion(self, measurement, human_sensor):
        """Performs fusion once per pass."""
        
        measurement_label = measurement['relation']
        relation_class = measurement['relation class']
        grounding = measurement['grounding']
        likelihood = grounding.relations.binary_models[relation_class]

        self.gm_fusion(likelihood, measurement_label, human_sensor)

        self.recently_fused_update = True

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

            # Perform fusion
            self.gm_fusion(likelihood, measurement_label, human_sensor)

            # Discard measurements for windowed, increase window size for full
            if self.fusion_method == 'windowed batch':
                self.measurements = []
            elif self.fusion_method == 'full batch':
                self.window += self.window

    def grid_fusion(self, measurement, human_sensor):
        if not hasattr(self, 'pos'):
            self._set_up_grid()

        if type(self.probability) is GaussianMixture:
            self.probability = self.probability.pdf(self.pos)

        prior_prob = self.probability

        measurement_label = measurement['relation']
        relation_class = measurement['relation class']
        grounding = measurement['grounding']
        likelihood = grounding.relations.binary_models[relation_class]
        likelihood_prob = likelihood.probability(class_=measurement_label, 
                                                 state=self.pos)

        posterior = likelihood_prob * prior_prob
        posterior /= posterior.sum()

        self.recently_fused_update = True

        self.probability = posterior


    def _set_up_grid(self, grid_size=0.1):
        bounds = self.feasible_layer.bounds
        self.X, self.Y = np.mgrid[bounds[0]:bounds[2]:grid_size,
                                  bounds[1]:bounds[3]:grid_size]
        self.pos = np.empty(self.X.shape + (2,))
        self.pos = np.dstack((self.X, self.Y))
        self.pos = np.reshape(self.pos, (self.X.size, 2))


    def gm_fusion(self, likelihood, measurement_label, human_sensor):
        
        if self.fusion_method == 'full batch':
            prior = self.original_prior
        else:
            prior = self.probability

        if type(likelihood) is list:
            mixtures = []
            raw_weights = []
            for u, mixand_weight in enumerate(prior.weights):

                prior_mixand = GaussianMixture(1, prior.means[u], prior.covariances[u])
                
                for i, geometric_sm in enumerate(likelihood):

                    mu, sigma, beta = self.vb.update(measurement=measurement_label,
                                                likelihood=geometric_sm,
                                                prior=prior_mixand,
                                                get_raw_beta=True,
                                                )
                    new_mixture = GaussianMixture(beta, mu, sigma)

                    # Weight the posterior by the human's false alarm rate
                    alpha = human_sensor.false_alarm_prob / 2
                    new_mixture = prior_mixand.combine_gms([new_mixture], alpha)

                    mixtures.append(new_mixture)
                    raw_weights.append(beta * mixand_weight)

            # Renormalize raw weights
            raw_weights = np.array(raw_weights).reshape(-1)
            raw_weights /= raw_weights.sum()

            try:
                posterior = mixtures[0].combine_gms(mixtures[1:], raw_weights=raw_weights)
            except IndexError:
                logging.error('ERROR! Cannot combine GMs.')
                posterior = prior

        else:
            mu, sigma, beta = self.vb.update(measurement=measurement_label,
                                             likelihood=likelihood,
                                             prior=prior,
                                             use_LWIS=False,
                                             )

            # Weight the posterior by the human's false alarm rate
            gm = GaussianMixture(beta, mu, sigma)
            alpha = human_sensor.false_alarm_prob / 2
            posterior = prior.combine_gms(gm, alpha)

        self.recently_fused_update = True

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
