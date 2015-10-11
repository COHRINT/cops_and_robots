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
                                                     uniform_prior,
                                                     )
from cops_and_robots.fusion.grid import Grid
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
                 fusion_method='grid',
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


        if self.fusion_method == 'grid':
            feasible_region = self.feasible_layer.pose_region
            prior = Grid(prior='fleming', feasible_region=feasible_region)
        else:
            prior = fleming_prior()
        self.probability = prior
        self.original_prior = prior

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

        self.probability.dynamics_update()
        # self.update_mixand_motion()
        self._camera_update(camera)
        self._human_update(human_sensor)
        # self.truncate_gaussians()

        if save_file is not None:
            if self.recently_fused_update:
                self.save_probability(save_file)
                self.recently_fused_update = False
            self.save_MAP(save_file)

    # def _camera_update(self, camera):
    #     mu, sigma, beta = self.vb.update(measurement='No Detection',
    #                                      likelihood=camera.detection_model,
    #                                      prior=self.probability,
    #                                      use_LWIS=True,
    #                                      poly=camera.detection_model.poly
    #                                      )
    #     bounds = self.feasible_layer.bounds
        
    #     try:
    #         pos = self.probability.pos
    #     except AttributeError:
    #         pos = None

    #     try:
    #         pos_all = self.probability.pos_all
    #     except AttributeError:
    #         pos_all = None
    #     gm = GaussianMixture(beta, mu, sigma, bounds=bounds, pos=pos, pos_all=pos_all)
    #     gm.camera_viewcone = camera.detection_model.poly  # for plotting
    #     self.probability = gm

    def _camera_update(self, camera):
        likelihood = camera.detection_model
        measurement = 'No Detection'

        self.probability.measurement_update(likelihood, measurement)
        self.probability.camera_viewcone = camera.detection_model.poly  # for plotting
        self.recently_fused_update = True

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

        self.fusion(likelihood, measurement_label, human_sensor)

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
            self.fusion(likelihood, measurement_label, human_sensor)

            # Discard measurements for windowed, increase window size for full
            if self.fusion_method == 'windowed batch':
                self.measurements = []
            elif self.fusion_method == 'full batch':
                self.window += self.window

    def grid_fusion(self, measurement, human_sensor):
        measurement_label = measurement['relation']
        relation_class = measurement['relation class']
        grounding = measurement['grounding']
        likelihood = grounding.relations.binary_models[relation_class]

        self.probability.measurment_update(likelihood_prob, measurement)

        self.recently_fused_update = True


    def fusion(self, likelihood, measurement, human_sensor):
        # prior = self.probability.copy()
        self.probability.measurment_update(likelihood, measurement)

        #<>TODO: include human false alarm rate
        self.recently_fused_update = True
        self.recieved_human_update = True


    def fusion_OLD(self, likelihood, measurement_label, human_sensor):
        
        if self.fusion_method == 'full batch':
            prior = self.original_prior
        else:
            prior = self.probability

        if type(likelihood) is list:
            # <>TODO: clean up this section!
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
            bounds  = self.feasible_layer.bounds
            try:
                pos = prior.pos
            except AttributeError:
                pos = None
            try:
                pos_all = prior.pos_all
            except AttributeError:
                pos_all = None
            gm = GaussianMixture(beta, mu, sigma, bounds=bounds, pos=pos, pos_all=pos_all)
            logging.info(gm)
            alpha = human_sensor.false_alarm_prob / 2
            posterior = prior.combine_gms(gm, alpha)

        self.recently_fused_update = True

        if posterior is not None:
            self.probability = posterior
            self.recieved_human_update = True

    def save_probability(self, save_file):
        if self.fusion_method == 'windowed batch':
            save_file = save_file + 'windowed_batch_' + str(self.window)
        else:
            save_file = save_file + self.fusion_method.replace(' ', '_')
        filename = save_file + '_' + self.target_name  +'_posterior_' + str(self.human_measurement_count)
        np.save(filename, self.probability)

    def save_MAP(self, save_file):
        self.frame += 1
        MAP_point, _ = self.probability.find_MAP()

        if self.fusion_method == 'windowed batch':
            save_file = save_file + 'windowed_batch_' + str(self.window)
        else:
            save_file = save_file + self.fusion_method.replace(' ', '_')
        filename = save_file + '_' + self.target_name  +'_MAP_' + str(self.frame)
        np.save(filename, MAP_point)

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
        dt = 0.1
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        Gamma = np.array([[0.5*dt**2, 0],
                          [0, 0.5*dt**2],
                          [dt, 0],
                          [0, dt]])
        Q = np.array([[.05, 0],
                      [0, .05]])

        weights = self.probability.weights
        means = self.probability.means
        # means = F*means
        covariances = self.probability.covariances
        for i, covariance in enumerate(covariances):
            covariances[i] = np.dot(np.dot(F, covariance), np.transpose(F)) + \
                np.dot(np.dot(Gamma, Q), np.transpose(Gamma))

        bounds = self.probability.bounds
        try:
            pos = self.probability.pos
        except AttributeError:
            pos = None

        try:
            pos_all = self.probability.pos_all
        except AttributeError:
            pos_all = None
        self.probability = GaussianMixture(weights=weights, means=means, covariances=covariances, bounds=bounds, pos=pos, pos_all=pos_all)




    def truncate_gaussians(self):
        # To start, just use map bounds
        bounds = self.feasible_layer.bounds
        logging.debug('Constraints: {}'.format(bounds))

        weights = self.probability.weights
        means = self.probability.means
        covariances = self.probability.covariances

        # V = np.array([[bounds[0],bounds[2]],[bounds[1],bounds[3]]])
        # Bcon, upper_bound = vert2con(V.T)

        Bcon = np.array([[1/bounds[0], 0, 0, 0],
                         [1/bounds[2], 0, 0, 0],
                         [0, 1/bounds[1], 0, 0],
                         [0, 1/bounds[3], 0, 0],
                         # [0, 0, 1, 1],
                         # [0, 0, -1, -1,],
                         ])
        upper_bound = np.array([[1],
                                [1],
                                [1],
                                [1],
                                # [1],
                                # [1],
                                ])
        lower_bound = -np.inf*np.ones((4, 1))

        new_means = []
        new_covariances = []
        for i, mean in enumerate(means):
            covariance = covariances[i]
            new_mean, new_covariance, wt = self.iterative_gaussian_trunc_update(mean,
                                              covariance, Bcon, lower_bound, upper_bound)
            new_means.append(new_mean)
            new_covariances.append(new_covariance)

        self.probability = GaussianMixture(weights=weights, means=new_means,
                                           covariances=new_covariances)


    def vert2con(V):
        # will assume convhull
        pass


    def iterative_gaussian_trunc_update(self, mean, covariance, Bcon,
                                        lower_bound, upper_bound,
                                        dosort=False, dosplit=False):
        if dosplit:
            pass

        if dosort:
            pass
            # probreductionmeasure = np.zeros(upperbound.shape)
            # for ii in range(Bcon):
            #     probreductionmeasure[ii] = (upperbound[ii]-Bcon[ii]*mean) / \
            #     np.sqrt(Bcon[ii] .dot covariance .dot Bcon[ii].T)
        else:
            Bmat = Bcon
            ubound = upper_bound
            lbound = lower_bound

        # Initialize mean and covariance matrix to be updated
        muTilde = mean
        SigmaTilde = covariance
        # print SigmaTilde

        # do iterative constraint updates
        for ii in range(Bmat.shape[0]):
            phi_ii = Bmat[ii].T

            # Eigenvalue decomp
            Tii, Wii = np.linalg.eig(SigmaTilde)
            # Take real parts
            Tii = np.real(Tii)
            Wii = np.real(Wii)
            # Make a diagonal matrix
            Tii = np.diag(Tii)
            
            # print 'eigenvector', Wii
            # print np.sqrt(Wii)
            # print 'Eigenvalues', Tii.T
            # print phi_ii

            # Find orthonogonal Sii via Gram-Schmidt
            P = np.sqrt(Wii) .dot (Tii.T) .dot (phi_ii)
            P = np.expand_dims(P, axis=0)

            # print 'P', P

            tau_ii = np.sqrt(phi_ii.T .dot (SigmaTilde) .dot (phi_ii))
            Qtilde, Rtilde = np.linalg.qr(P)

            # print 'R', Rtilde
            # print tau_ii
            # print Qtilde
            # Sii = (Rtilde[0][0] / tau_ii) * (Qtilde.T)

            # Compute transformed lower and upper 1D constraint bounds
            # print 'mu', muTilde
            # print 'phi', phi_ii
            # print phi_ii.T .dot (muTilde)
            # print lbound[ii]
            cii = (lbound[ii] - phi_ii.T .dot (muTilde)) / tau_ii
            dii = (ubound[ii] - phi_ii.T .dot (muTilde)) / tau_ii

            print 'cii', cii
            print 'dii', dii

            # compute renormalization stats
            alphaiiden = np.maximum(sp.special.erf(dii/np.sqrt(2)) - sp.special.erf(cii/np.sqrt(2)), np.finfo(float).eps)
            alphaii = np.sqrt(2/np.pi) / alphaiiden
            muii = alphaii * np.exp(-0.5 * cii ** 2) - np.exp(-0.5 * dii ** 2)

            # check for -/+ inf bounds to avoid nans
            if np.isinf(cii).all() and not np.isinf(dii).all():
                sig2ii = alphaii * ( -np.exp(-0.5*dii ** 2) * (dii-2*muii) ) + muii ** 2 + 1
            elif np.isinf(dii).all() and not np.isinf(cii).all():
                sig2ii = alphaii * ( np.exp(-0.5*cii ** 2) * (cii-2*muii) ) + muii ** 2 + 1
            elif np.isinf(dii).all() and np.isinf(cii).all():
                sig2ii = muii ** 2 + 1
            else:
                sig2ii = alphaii * ( np.exp(-0.5*cii ** 2)*(cii-2*muii) - \
                    np.exp(-0.5*dii ** 2)*(dii-2*muii) ) + muii ** 2 + 1

            if sig2ii <= 0:
                logging.error('Something''s wrong: sig2ii <=0!') 

            # get mean and covariance of transformed state estimate:
            ztilde_ii = np.concatenate((np.expand_dims(muii, axis=0), np.zeros((muTilde.shape[0]-1, 1))), axis=0)
            Ctilde_ii = np.diag(np.concatenate((np.expand_dims(sig2ii, axis=0), np.ones((muTilde.shape[0]-1,1)))));

            # recover updated estimate in original state space for next/final pass
            muTilde = Tii * np.sqrt(Wii) * Sii.T * ztilde_ii + muTilde
            SigmaTilde = Tii * np.sqrt(Wii)*Sii.T * Ctilde_ii * Sii * np.sqrt(Wii) * Tii.T
            print Tii
            print Wii
            print 'Sii', Sii.T
            print Ctilde_ii

            # ensure symmetry:
            SigmaTilde = 0.5 * (SigmaTilde + SigmaTilde.T)
            print SigmaTilde

        muOut = muTilde
        SigmaOut = SigmaTilde
        # compute updated likelihood
        # pass
        wtOut = 1

        return muOut, SigmaOut, wtOut #lkOut

