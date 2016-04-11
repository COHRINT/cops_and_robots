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
from cops_and_robots.fusion.filter import Filter
from cops_and_robots.fusion.variational_bayes import VariationalBayes
from cops_and_robots.fusion.softmax import (geometric_model,
                                            neighbourhood_model,
                                            product_model, 
                                            )

class GaussSumFilter(Filter):
    """docstring for GaussSumFilter

    Fusion methods describe how to perform data fusion, with sequential updating
    at each time step, full batch doing a complete batch update of all sensor
    information from the initial prior, and windowed batch fusing all sensor
    information provided within a specific window.

    Compression methods describe how a batch (full or windowed) fusion is
    performed. Product is exact fusion, neighbourhood uses a reduced number
    of neighbour classes near the joint measurement class, and geometric uses
    the minimum number of classes next to the joint measurement class.

    """
    fusion_methods = ['sequential', 'full batch', 'windowed batch']
    compression_methods = ['product', 'neighbourhood', 'geometric']

    def __init__(self, 
                 fusion_method='sequential',
                 compression_method='geometric',
                 window=1,
                 *args,
                 **kwargs
                 ):
        super(GaussSumFilter, self).__init__(probability_type='gaussian_mixture',
                                             *args, **kwargs)

        self.fusion_method = fusion_method
        self.compression_method = compression_method
        self.window = window

        # Set up the VB fusion parameters
        self.vb = VariationalBayes()

    def _human_update(self, human_sensor):

        # Validate human sensor statement
        measurement = self._verify_human_update(human_sensor)
        if measurement is None:
            return

        if self.rosbag_process is not None:
            logging.info('Stopped rosbag to do fusion...')
            self.rosbag_process.stdin.write(' ')  # stop 
            self.rosbag_process.stdin.flush()
            time.sleep(0.5)

        if self.fusion_method == 'sequential':
            self.sequential_fusion(measurement, human_sensor)
        elif self.fusion_method == 'full batch' \
            or self.fusion_method == 'windowed batch':
            self.batch_fusion(measurement, human_sensor)
        elif self.fusion_method == 'grid':
            self.sequential_fusion(measurement, human_sensor)

        if self.rosbag_process is not None:
            self.rosbag_process.stdin.write(' ')  # start rosbag
            self.rosbag_process.stdin.flush()
            logging.info('Restarted rosbag!')

    def sequential_fusion(self, measurement, human_sensor):
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

            # Concatenate softmax models
            models = []
            for measurement in self.measurements:
                grounding = measurement['grounding']
                relation_class = measurement['relation class']
                model = grounding.relations.binary_models[relation_class]
                models.append(model)

            # Compress the likelihood
            if self.compression_method == 'product':
                likelihood = product_model(models)
            elif self.compression_method == 'neighbourhood':
                likelihood = neighbourhood_model(models, measurement_labels)
            elif self.compression_method == 'geometric':
                likelihood = geometric_model(models, measurement_labels)

            # Perform fusion
            self.fusion(likelihood, measurement_label, human_sensor)

            # Discard measurements for windowed, increase window size for full
            if self.fusion_method == 'windowed batch':
                self.measurements = []
            elif self.fusion_method == 'full batch':
                self.window += self.window


    def fusion(self, likelihood, measurement, human_sensor):

        if type(likelihood) is list:
            self.multi_likelihood_fusion(likelihood, measurement, human_sensor)
        else:
            self.probability.measurement_update(likelihood, measurement)

        #<>TODO: include human false alarm rate
        self.recently_fused_update = True
        self.recieved_human_update = True


    def multi_likelihood_fusion(self, likelihoods, measurement_label, human_sensor):
        
        if self.fusion_method == 'full batch':
            prior = self.original_prior
        else:
            prior = self.probability

        # <>TODO: clean up this section!
        mixtures = []
        raw_weights = []
        for u, mixand_weight in enumerate(prior.weights):

            prior_mixand = GaussianMixture(1, prior.means[u], prior.covariances[u])

            for i, likelihood in enumerate(likelihoods):

                mu, sigma, beta = self.vb.update(measurement=measurement_label,
                                                 likelihood=likelihood,
                                                 prior=prior_mixand,
                                                 get_raw_beta=True,
                                                 )
                new_mixture = GaussianMixture(beta, mu, sigma)

                # Weight the posterior by the human's false alarm rate
                alpha = human_sensor.false_alarm_prob / 2
                prior_mixand.combine_gms([new_mixture], alpha)

                mixtures.append(prior_mixand)
                raw_weights.append(beta * mixand_weight)

        # Renormalize raw weights
        raw_weights = np.array(raw_weights).reshape(-1)
        raw_weights /= raw_weights.sum()

        try:
            mixtures[0].combine_gms(mixtures[1:], raw_weights=raw_weights)
            posterior = mixtures[0]
        except IndexError:
            logging.error('ERROR! Cannot combine GMs.')
            posterior = prior
        self.probability = posterior

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


def test_fusion(fusion_method='sequential', speed_test=True):
    from cops_and_robots.map_tools.map import Map
    from cops_and_robots.map_tools.probability_layer import ProbabilityLayer
    from cops_and_robots.human_tools.human import Human
    import matplotlib.pyplot as plt

    map_ = Map()
    human_sensor = Human(map_=map_)

    kwargs = {'target_name': 'Roy',
              'fusion_method': fusion_method,
              'dynamic_model': False,
              }
    product_filter = GaussSumFilter(compression_method='product', **kwargs)
    neighbourhood_filter = GaussSumFilter(compression_method='neighbourhood', **kwargs)
    geometric_filter = GaussSumFilter(compression_method='geometric', **kwargs)

    # Plot initial state
    fig = plt.Figure()
    probability_layer = ProbabilityLayer(geometric_filter, bounds=map_.bounds,
                                         grid_size=0.1, fig=fig)
    # probability_layer.plot()
    # plt.show()

    # Plot sensor updates
    human_utterances = ['I know Roy is inside the hallway.',
                        'I know Roy is near the fern.',
                        'I know Roy is not inside the kitchen.',
                        ]
    for utterance in human_utterances:
        human_sensor.utterance = utterance
        human_sensor.new_update = True
        geometric_filter.update(human_sensor=human_sensor)

        fig = plt.Figure()
        probability_layer = ProbabilityLayer(geometric_filter, bounds=map_.bounds,
                                             grid_size=0.1, fig=fig)
        # probability_layer.plot()
        # plt.show()


if __name__ == '__main__':
    test_fusion()


