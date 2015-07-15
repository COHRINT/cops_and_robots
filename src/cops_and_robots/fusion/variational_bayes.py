#!/usr/bin/env python
from __future__ import division
"""Variational Bayes model to fuse

DESCRIPTION

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

import numpy as np
from numpy.linalg import inv, det
import matplotlib.pyplot as plt

from cops_and_robots.fusion.softmax import (speed_model,
                                            intrinsic_space_model,
                                            binary_speed_model,
                                            range_model,
                                            binary_range_model,
                                            camera_model_2D)


from cops_and_robots.fusion.gaussian_mixture import GaussianMixture


class VariationalBayes(object):
    """short description of VariationalBayes

    long description of VariationalBayes

    Parameters
    ----------
    param : param_type, optional
        param_description

    Attributes
    ----------
    attr : attr_type
        attr_description

    Methods
    ----------
    attr : attr_type
        attr_description

    """

    def __init__(self,
                 num_EM_convergence_loops=15,
                 EM_convergence_tolerance=10 ** -3,
                 max_EM_steps=250,
                 num_importance_samples=500,
                 num_mixand_samples=500,
                 weight_threshold=None,
                 mix_sm_corr_thresh=0.95,
                 max_num_mixands=50):
        self.num_EM_convergence_loops = num_EM_convergence_loops
        self.EM_convergence_tolerance = EM_convergence_tolerance
        self.max_EM_steps = max_EM_steps
        self.num_importance_samples = num_importance_samples
        self.num_mixand_samples = num_mixand_samples
        if weight_threshold is None:
            weight_threshold = np.finfo(float).eps  # Machine precision
        self.weight_threshold = weight_threshold
        self.mix_sm_corr_thresh = mix_sm_corr_thresh
        self.max_num_mixands = max_num_mixands

    def vb_update(self, measurement, likelihood, prior,
                  init_mean=0, init_var=1, init_alpha=0.5, init_xi=1):
        """Variational bayes update for Gaussian and Softmax.
        """
        # Likelihood values
        if hasattr(likelihood, 'subclasses'):
            m = likelihood.num_subclasses
            j = likelihood.subclasses[measurement].id
            init_xi = np.ones(likelihood.num_subclasses)
        else:
            m = likelihood.num_classes
            j = likelihood.classes[measurement].id
            init_xi = np.ones(likelihood.num_classes)
        w = likelihood.weights
        b = likelihood.biases


        xis, alpha, mu_hat, var_hat, prior_mean, prior_var = \
            self._check_inputs(likelihood, init_mean, init_var, init_alpha, init_xi, prior)


        converged = False
        EM_step = 0

        while not converged and EM_step < self.max_EM_steps:
            ################################################################
            # STEP 1 - EXPECTATION
            ################################################################
            # PART A #######################################################

            # find g_j
            sum1 = 0
            for c in range(m):
                if c != j:
                    sum1 += b[c]
            sum2 = 0
            for c in range(m):
                sum2 = xis[c] / 2 \
                    + self._lambda(xis[c]) * (xis[c] ** 2 - (b[c] - alpha) ** 2) \
                    - np.log(1 + np.exp(xis[c]))
            g_j = 0.5 * (b[j] - sum1) + alpha * (m / 2 - 1) + sum2

            # find h_j
            sum1 = 0
            for c in range(m):
                if c != j:
                    sum1 += w[c]
            sum2 = 0
            for c in range(m):
                sum2 += self._lambda(xis[c]) * (alpha - b[c]) * w[c]
            h_j = 0.5 * (w[j] - sum1) + 2 * sum2

            # find K_j
            sum1 = 0
            for c in range(m):
                sum1 += self._lambda(xis[c]) * np.outer(w[c], (w[c]))

            K_j = 2 * sum1

            K_p = inv(prior_var)
            g_p = -0.5 * (np.log(np.linalg.det(2 * np.pi * prior_var))) \
                + prior_mean.T .dot (K_p) .dot (prior_var)
            h_p = K_p .dot (prior_mean)

            g_l = g_p + g_j
            h_l = h_p + h_j
            K_l = K_p + K_j

            mu_hat = inv(K_l) .dot (h_l)
            var_hat = inv(K_l)

            # PART B #######################################################
            y_cs = np.zeros(m)
            y_cs_squared = np.zeros(m)
            for c in range(m):
                y_cs[c] = w[c].T .dot (mu_hat) + b[c]
                y_cs_squared[c] = w[c].T .dot \
                    (var_hat + np.outer(mu_hat, mu_hat.T)) .dot (w[c]) \
                    + 2 * w[c].T .dot (mu_hat) * b[c] + b[c] ** 2

            ################################################################
            # STEP 2 - MAXIMIZATION
            ################################################################
            for i in range(self.num_EM_convergence_loops):  # n_{lc}

                # PART A ######################################################
                # Find xis
                for c in range(m):
                    xis[c] = np.sqrt(y_cs_squared[c] + alpha ** 2 - 2 * alpha
                                     * y_cs[c])

                # PART B ######################################################
                # Find alpha
                num_sum = 0
                den_sum = 0
                for c in range(m):
                    num_sum += self._lambda(xis[c]) * y_cs[c]
                    den_sum += self._lambda(xis[c])
                alpha = ((m - 2) / 4 + num_sum) / den_sum

            ################################################################
            # STEP 3 - CONVERGENCE CHECK
            ################################################################
            if EM_step == 0:
                prev_log_c_hat = -1000  # Arbitrary value

            KLD = 0.5 * (np.log(det(prior_var) / det(var_hat)) +
                         np.trace(inv(prior_var) .dot (var_hat)) +
                         (prior_mean - mu_hat).T .dot (inv(prior_var)) .dot
                         (prior_mean - mu_hat))

            sum1 = 0
            for c in range(m):
                sum1 += 0.5 * (alpha + xis[c] - y_cs[c]) \
                    - self._lambda(xis[c]) * (y_cs_squared[c] - 2 * alpha
                    * y_cs[c] + alpha ** 2 - xis[c] ** 2) \
                    - np.log(1 + np.exp(xis[c]))

            # <>TODO: don't forget Mun - unobserved parents!
            # <>CHECK - WHY DO WE ADD +1 HERE??
            log_c_hat = y_cs[j] - alpha + sum1 - KLD + 1

            if np.abs(log_c_hat - prev_log_c_hat) < self.EM_convergence_tolerance:
                logging.debug('Convergence reached at step {} with log_c_hat {}'
                              .format(EM_step, log_c_hat))
                break

            prev_log_c_hat = log_c_hat
            EM_step += 1

        # Resize parameters
        if mu_hat.size == 1:
            mu_post = mu_hat[0]
        else:
            mu_post = mu_hat
        if var_hat.size == 1:
            var_post = var_hat[0][0]
        else:
            var_post = var_hat

        logging.debug('VB update found mean of {} and variance of {}.'
                     .format(mu_post, var_post))

        return mu_post, var_post, log_c_hat

    def lwis_update(self, prior):
        """

        clustering:
            pairwise greedy merging - compare means, weights & variances
            salmond's method and runnals' method (better)

        """
        prior_mean = np.asarray(prior.means[0])
        prior_var = np.asarray(prior.covariances[0])

        # Importance distribution
        q = GaussianMixture(1, prior_mean, prior_var)

        # Importance sampling correction
        w = np.zeros(num_samples)  # Importance weights
        x = q.rvs(size=num_samples)  # Sampled points
        x = np.asarray(x)
        if hasattr(likelihood, 'subclasses'):
            measurement_class = likelihood.subclasses[measurement]
        else:
            measurement_class = likelihood.classes[measurement]

        for i in range(num_samples):
            w[i] = prior.pdf(x[i]) \
                * measurement_class.probability(state=x[i])\
                / q.pdf(x[i])
        w /= np.sum(w)  # Normalize weights

        mu_hat = np.zeros_like(np.asarray(mu_VB))
        for i in range(num_samples):
            x_i = np.asarray(x[i])
            mu_hat = mu_hat + x_i .dot (w[i])

        var_hat = np.zeros_like(np.asarray(var_VB))
        for i in range(num_samples):
            x_i = np.asarray(x[i])
            var_hat = var_hat + w[i] * np.outer(x_i, x_i) 
        var_hat -= np.outer(mu_hat, mu_hat)

        if mu_hat.size == 1 and mu_hat.ndim > 0:
            mu_lwis = mu_hat[0]
        else:
            mu_lwis = mu_hat
        if var_hat.size == 1:
            var_lwis = var_hat[0][0]
        else:
            var_lwis = var_hat

        logging.debug('LWIS update found mean of {} and variance of {}.'
                     .format(mu_lwis, var_lwis))

        return mu_lwis, var_lwis, log_c_hat


    def vbis_update(self, measurement, likelihood, prior,
                    init_mean=0, init_var=1, init_alpha=0.5, init_xi=1,
                    num_samples=None,  use_LWIS=False):
        """VB update with importance sampling for Gaussian and Softmax.
        """
        if num_samples is None:
            num_samples = self.num_importance_samples

        if use_LWIS:
            q_mu = np.asarray(prior.means[0])
            log_c_hat = np.nan
        else:
            # Use VB update
            q_mu, var_VB, log_c_hat = self.vb_update(measurement, likelihood,
                                                      prior,
                                                      init_mean, init_var,
                                                      init_alpha, init_xi)

        q_var = np.asarray(prior.covariances[0])

        # Importance distribution
        q = GaussianMixture(1, q_mu, q_var)

        # Importance sampling correction
        w = np.zeros(num_samples)  # Importance weights
        x = q.rvs(size=num_samples)  # Sampled points
        x = np.asarray(x)
        if hasattr(likelihood, 'subclasses'):
            measurement_class = likelihood.subclasses[measurement]
        else:
            measurement_class = likelihood.classes[measurement]

        # Compute parameters using samples
        w = prior.pdf(x) * measurement_class.probability(state=x) / q.pdf(x)
        w /= np.sum(w)  # Normalize weights

        mu_hat = np.sum(x.T * w, axis=-1)

        # <>TODO: optimize this
        var_hat = np.zeros_like(np.asarray(q_var))
        for i in range(num_samples):
            x_i = np.asarray(x[i])
            var_hat = var_hat + w[i] * np.outer(x_i, x_i) 
        var_hat -= np.outer(mu_hat, mu_hat)

        # Ensure properly formatted output
        if mu_hat.size == 1 and mu_hat.ndim > 0:
            mu_post_vbis = mu_hat[0]
        else:
            mu_post_vbis = mu_hat
        if var_hat.size == 1:
            var_post_vbis = var_hat[0][0]
        else:
            var_post_vbis = var_hat

        logging.debug('VBIS update found mean of {} and variance of {}.'
                     .format(mu_post_vbis, var_post_vbis))

        return mu_post_vbis, var_post_vbis, log_c_hat

    def update(self, measurement, likelihood, prior, use_LWIS=False):
        """VB update using Gaussian mixtures and multimodal softmax.

        This uses Variational Bayes with Importance Sampling (VBIS) for
        each mixand-softmax pair available.
        """

        h = 0
        relevant_subclasses = likelihood.classes[measurement].subclasses
        num_relevant_subclasses = len(relevant_subclasses)

        # Parameters for all new mixands
        K = num_relevant_subclasses * prior.weights.size
        mu_hat = np.zeros((K, prior.means.shape[1]))
        var_hat = np.zeros((K, prior.covariances.shape[1],
                            prior.covariances.shape[2]))
        log_beta_hat = np.zeros(K) # Weight estimates

        for u, mixand_weight in enumerate(prior.weights):
            mix_sm_corr = 0
            
            # Check to see if the mixand is completely contained within
            # the softmax subclasses (i.e. doesn't need an update)

            # <>TODO: check prob of class instead of looping over subclasses
            for label, subclass in relevant_subclasses.iteritems():
                # Compute \hat{P}_s(r|u)
                mixand = GaussianMixture(1, prior.means[u],
                                          prior.covariances[u])
                mixand_samples = mixand.rvs(self.num_mixand_samples)
                p_hat_ru_samples = subclass.probability(state=mixand_samples)
                p_hat_ru_sampled = np.sum(p_hat_ru_samples) / self.num_mixand_samples
                mix_sm_corr += p_hat_ru_sampled

            if mix_sm_corr > self.mix_sm_corr_thresh:
                logging.debug('Mixand {}\'s correspondence with {}\'s subclasses'
                             ' was {}, above the threshold of {}, so VBIS was '
                             'skipped.'.format(u, measurement, mix_sm_corr,
                                               self.mix_sm_corr_thresh))

                # Append the prior's parameters to the mixand parameter lists
                mu_hat[h, :] = prior.means[u]
                var_hat[h, :] = prior.covariances[u]
                log_beta_hat[h] = np.log(mixand_weight)

                h +=1
                continue

            # Otherwise complete the full VBIS update
            ordered_subclasses = iter(sorted(relevant_subclasses.iteritems()))
            for label, subclass in ordered_subclasses:

                # Compute \hat{P}_s(r|u)
                mixand = GaussianMixture(1, prior.means[u],
                                         prior.covariances[u])
                mixand_samples = mixand.rvs(self.num_mixand_samples)
                p_hat_ru_samples = subclass.probability(state=mixand_samples)
                p_hat_ru_sampled = np.sum(p_hat_ru_samples) / self.num_mixand_samples

                mu_vbis, var_vbis, log_c_hat = \
                    self.vbis_update(label, subclass.softmax_collection,
                                     mixand, use_LWIS=use_LWIS)

                # Compute log odds of r given u
                if np.isnan(log_c_hat):  # from LWIS update
                    log_p_hat_ru = np.log(p_hat_ru_sampled)
                else:
                    log_p_hat_ru = np.max((log_c_hat, np.log(p_hat_ru_sampled)))

                # Find log of P(u,r|D_k) \approxequal \hat{B}_{ur}
                log_beta_vbis = np.log(mixand_weight) + log_p_hat_ru

                # Symmetrize var_vbis
                var_vbis = 0.5 * (var_vbis.T + var_vbis)

                # 

                # Update estimate values
                log_beta_hat[h] = log_beta_vbis
                mu_hat[h,:] = mu_vbis
                var_hat[h,:] = var_vbis
                h += 1

        # Renormalize and truncate (based on weight threshold)
        log_beta_hat = log_beta_hat - np.max(log_beta_hat)
        beta_hat = np.exp(log_beta_hat) / np.sum(np.exp(log_beta_hat))

        mu_hat = mu_hat[beta_hat > self.weight_threshold, :]
        var_hat = var_hat[beta_hat > self.weight_threshold, :]
        beta_hat = beta_hat[beta_hat > self.weight_threshold]

        # Shrink mu, var and beta if necessary
        beta_hat = beta_hat[:h]
        mu_hat = mu_hat[:h]
        var_hat = var_hat[:h]

        return mu_hat, var_hat, beta_hat

    def _lambda(self, xi_c):
        return 1 / (2 * xi_c) * ( (1 / (1 + np.exp(-xi_c))) - 0.5)


    def _check_inputs(self, likelihood, init_mean, init_var, init_alpha, init_xi, prior):
        # Make sure inputs are numpy arrays
        init_mean = np.asarray(init_mean)
        init_var = np.asarray(init_var)
        init_alpha = np.asarray(init_alpha)
        init_xi = np.asarray(init_xi)

        if init_xi.ndim != 1:
            try:
                m = likelihood.num_subclasses
                assert init_xi.size == m
            except AssertionError:
                logging.exception('Initial xi was not the right size.')
                raise
            init_xi = np.reshape(init_xi, (1, -1))
            logging.debug("Initial xi is not the right shape. Reshaping.")

        # Preparation
        xis = init_xi
        alpha = init_alpha
        mu_hat = init_mean
        var_hat = init_var

        # <>EXTEND
        prior_mean = prior.means[0]
        prior_var = prior.covariances[0]

        return xis, alpha, mu_hat, var_hat, prior_mean, prior_var


def comparison_1d():

    # Define prior 
    prior_mean, prior_var = 0.3, 0.01
    min_x, max_x = -5, 5
    res = 10000

    prior = GaussianMixture(1, prior_mean, prior_var)
    x_space = np.linspace(min_x, max_x, res)

    # Define sensor likelihood
    sm = speed_model()
    measurement = 'Slow'
    measurement_i = sm.class_labels.index(measurement)

    # Do a VB update
    init_mean, init_var = 0, 1
    init_alpha, init_xi = 0.5, np.ones(4)

    vb = VariationalBayes()
    vb_mean, vb_var, _ = vb.vb_update(measurement, sm, prior, init_mean,
                                       init_var, init_alpha, init_xi)
    vb_posterior = GaussianMixture(1, vb_mean, vb_var)

    nisar_vb_mean = 0.131005297841171
    nisar_vb_var = 6.43335516254277e-05
    diff_vb_mean = vb_mean - nisar_vb_mean
    diff_vb_var = vb_var - nisar_vb_var
    logging.info('Nisar\'s VB update had mean difference {} and var difference {}\n'
                 .format(diff_vb_mean, diff_vb_var))

    # Do a VBIS update
    vbis_mean, vbis_var, _ = vb.vbis_update(measurement, sm, prior, init_mean,
                                         init_var, init_alpha, init_xi)
    vbis_posterior = GaussianMixture(1, vbis_mean, vbis_var)

    nisar_vbis_mean = 0.154223416817080
    nisar_vbis_var = 0.00346064073274943
    diff_vbis_mean = vbis_mean - nisar_vbis_mean
    diff_vbis_var = vbis_var - nisar_vbis_var
    logging.info('Nisar\'s VBIS update had mean difference {} and var difference {}\n'
                 .format(diff_vbis_mean, diff_vbis_var))

    # Plot results
    likelihood_label = 'Likelihood of \'{}\''.format(measurement)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sm.classes[measurement].plot(ax=ax, fill_between=False, label=likelihood_label, ls='--')
    ax.plot(x_space, prior.pdf(x_space), lw=1, label='prior pdf', c='grey', ls='--')

    ax.plot(x_space, vb_posterior.pdf(x_space), lw=2, label='VB posterior', c='r')
    ax.fill_between(x_space, 0, vb_posterior.pdf(x_space), alpha=0.2, facecolor='r')
    ax.plot(x_space, vbis_posterior.pdf(x_space), lw=2, label='VBIS Posterior', c='g')
    ax.fill_between(x_space, 0, vbis_posterior.pdf(x_space), alpha=0.2, facecolor='g')

    ax.set_title('VBIS Update')
    ax.legend()
    ax.set_xlim([0, 0.4])
    ax.set_ylim([0, 7])
    plt.show()


def comparison_2d():
    # Define prior 
    prior_mean = np.array([2.3, 1.2])
    prior_var = np.array([[2, 0.6], [0.6, 2]])
    prior = GaussianMixture(1, prior_mean, prior_var)

    # Define sensor likelihood
    sm = intrinsic_space_model()
    measurement = 'Front'
    measurement_i = sm.classes[measurement].id

    # Do a VB update
    init_mean = np.zeros((1,2))
    init_var = np.eye(2)
    init_alpha = 0.5
    init_xi = np.ones(5)

    vb = VariationalBayes()
    vb_mean, vb_var, _ = vb.vb_update(measurement, sm, prior, init_mean,
                                       init_var, init_alpha, init_xi)

    nisar_vb_mean = np.array([1.795546121012238, 2.512627005425541])
    nisar_vb_var = np.array([[0.755723395661314, 0.091742424424428],
                             [0.091742424424428, 0.747611340151417]])
    diff_vb_mean = vb_mean - nisar_vb_mean
    diff_vb_var = vb_var - nisar_vb_var
    logging.info('Nisar\'s VB update had mean difference: \n {}\n and var difference: \n {}\n'
                 .format(diff_vb_mean, diff_vb_var))

    vb_mean, vb_var, _ = vb.vbis_update(measurement, sm, prior, init_mean,
                                       init_var, init_alpha, init_xi)
    vb_posterior = GaussianMixture(1, vb_mean, vb_var)

    # Define gridded space for graphing
    min_x, max_x = -5, 5
    min_y, max_y = -5, 5
    res = 200
    x_space, y_space = np.mgrid[min_x:max_x:1/res,
                                min_y:max_y:1/res]
    pos = np.empty(x_space.shape + (2,))
    pos[:, :, 0] = x_space; pos[:, :, 1] = y_space;

    levels_res = 30
    max_prior = np.max(prior.pdf(pos))
    prior_levels = np.linspace(0, max_prior, levels_res)
    
    sm.probability()
    max_lh = np.max(sm.probs)
    lh_levels = np.linspace(0, max_lh, levels_res)
    
    max_post = np.max(vb_posterior.pdf(pos))
    post_levels = np.linspace(0, max_post, levels_res)

    # Plot results
    fig = plt.figure()
    likelihood_label = 'Likelihood of \'{}\''.format(measurement)
    
    prior_ax = plt.subplot2grid((2,32), (0,0), colspan=14)
    prior_cax = plt.subplot2grid((2,32), (0,14), colspan=1)
    prior_c = prior_ax.contourf(x_space, y_space, prior.pdf(pos), levels=prior_levels)
    cbar = plt.colorbar(prior_c, cax=prior_cax)
    prior_ax.set_xlabel('x1')
    prior_ax.set_ylabel('x2')
    prior_ax.set_title('Prior Distribution')

    lh_ax = plt.subplot2grid((2,32), (0,17), colspan=14)
    lh_cax = plt.subplot2grid((2,32), (0,31), colspan=1)
    sm.classes[measurement].plot(ax=lh_ax, label=likelihood_label, plot_3D=False, levels=lh_levels)
    # plt.colorbar(sm.probs, cax=lh_cax)
    lh_ax.set_title(likelihood_label)

    posterior_ax = plt.subplot2grid((2,32), (1,0), colspan=31)
    posterior_cax = plt.subplot2grid((2,32), (1,31), colspan=1)
    posterior_c = posterior_ax.contourf(x_space, y_space, vb_posterior.pdf(pos),  levels=post_levels)
    plt.colorbar(posterior_c, cax=posterior_cax)
    posterior_ax.set_xlabel('x1')
    posterior_ax.set_ylabel('x2')
    posterior_ax.set_title('VB Posterior Distribution')

    plt.show()


def gmm_sm_test(measurement='Outside'):

    # Define prior 
    # prior = GaussianMixture(weights=[1, 4, 5],
    #                         means=[[0.5, 1.3],  # GM1 mean
    #                                [-0.7, -0.6],  # GM2 mean
    #                                [0.2, -3],  # GM3 mean
    #                                ],
    #                         covariances=[[[0.4, 0.3],  # GM1 mean
    #                                       [0.3, 0.4]
    #                                       ],
    #                                      [[0.3, 0.1],  # GM2 mean
    #                                       [0.1, 0.3]
    #                                       ],
    #                                      [[0.5, 0.4],  # GM3 mean
    #                                       [0.4, 0.5]],
    #                                      ])
    prior = GaussianMixture(weights=[1, 1, 1, 1, 1],
                            means=[[-2, -4],  # GM1 mean
                                   [-1, -2],  # GM2 mean
                                   [0, 0],  # GM3 mean
                                   [1, -2],  # GM4 mean
                                   [2, -4],  # GM5 mean
                                   ],
                            covariances=[[[0.1, 0],  # GM1 mean
                                          [0, 0.1]
                                          ],
                                         [[0.2, 0],  # GM2 mean
                                          [0, 0.2]
                                          ],
                                         [[0.3, 0],  # GM3 mean
                                          [0, 0.3]
                                          ],
                                         [[0.2, 0],  # GM4 mean
                                          [0, 0.2]
                                          ],
                                         [[0.1, 0],  # GM5 mean
                                          [0, 0.1]],
                                         ])
    # prior = GaussianMixture(weights=[1],
    #                         means=[[-2, -4],  # GM1 mean
    #                                ],
    #                         covariances=[[[0.1, 0],  # GM1 mean
    #                                       [0, 0.1]
    #                                       ],
    #                                      ])
    # Define sensor likelihood
    brm = range_model()

    # Do a VBIS update
    logging.info('Starting VB update...')
    vb = VariationalBayes()
    mu_hat, var_hat, beta_hat = vb.update(measurement, brm, prior, use_LWIS=True)
    vbis_posterior = GaussianMixture(weights=beta_hat, means=mu_hat, covariances=var_hat)

    # Define gridded space for graphing
    min_x, max_x = -5, 5
    min_y, max_y = -5, 5
    res = 100
    x_space, y_space = np.mgrid[min_x:max_x:1/res,
                                min_y:max_y:1/res]
    pos = np.empty(x_space.shape + (2,))
    pos[:, :, 0] = x_space; pos[:, :, 1] = y_space;

    levels_res = 50
    max_prior = np.max(prior.pdf(pos))
    prior_levels = np.linspace(0, max_prior, levels_res)

    brm.probability()
    max_lh = np.max(brm.probs)
    lh_levels = np.linspace(0, max_lh, levels_res)
    max_post = np.max(vbis_posterior.pdf(pos))
    post_levels = np.linspace(0, max_post, levels_res)

    # Plot results
    fig = plt.figure()
    likelihood_label = 'Likelihood of \'{}\''.format(measurement)
    
    prior_ax = plt.subplot2grid((2,32), (0,0), colspan=14)
    prior_cax = plt.subplot2grid((2,32), (0,14), colspan=1)
    prior_c = prior_ax.contourf(x_space, y_space, prior.pdf(pos), levels=prior_levels)
    cbar = plt.colorbar(prior_c, cax=prior_cax)
    prior_ax.set_xlabel('x1')
    prior_ax.set_ylabel('x2')
    prior_ax.set_title('Prior Distribution')

    lh_ax = plt.subplot2grid((2,32), (0,17), colspan=14)
    lh_cax = plt.subplot2grid((2,32), (0,31), colspan=1)
    brm.classes[measurement].plot(ax=lh_ax, label=likelihood_label, ls='--', levels=lh_levels, show_plot=False, plot_3D=False)
    # plt.colorbar(sm.probs, cax=lh_cax)
    lh_ax.set_title(likelihood_label)

    posterior_ax = plt.subplot2grid((2,32), (1,0), colspan=31)
    posterior_cax = plt.subplot2grid((2,32), (1,31), colspan=1)
    posterior_c = posterior_ax.contourf(x_space, y_space, vbis_posterior.pdf(pos),  levels=post_levels)
    plt.colorbar(posterior_c, cax=posterior_cax)
    posterior_ax.set_xlabel('x1')
    posterior_ax.set_ylabel('x2')
    posterior_ax.set_title('VBIS Posterior Distribution')

    logging.info('Prior Weights: \n {} \n Means: \n {} \n Variances: \n {} \n'.format(prior.weights,prior.means,prior.covariances))
    logging.info('Posterior Weights: \n {} \n Means: \n {} \n Variances: \n {} \n'.format(vbis_posterior.weights,vbis_posterior.means,vbis_posterior.covariances))

    plt.show()


def compare_to_matlab(measurement='Near'):
    prior = GaussianMixture(weights=[1, 1, 1, 1, 1],
                            means=[[-2, -4],  # GM1 mean
                                   [-1, -2],  # GM2 mean
                                   [0, 0],  # GM3 mean
                                   [1, -2],  # GM4 mean
                                   [2, -4],  # GM5 mean
                                   ],
                            covariances=[[[0.1, 0],  # GM1 mean
                                          [0, 0.1]
                                          ],
                                         [[0.2, 0],  # GM2 mean
                                          [0, 0.2]
                                          ],
                                         [[0.3, 0],  # GM3 mean
                                          [0, 0.3]
                                          ],
                                         [[0.2, 0],  # GM4 mean
                                          [0, 0.2]
                                          ],
                                         [[0.1, 0],  # GM5 mean
                                          [0, 0.1]],
                                         ])

    # prior = GaussianMixture(weights=[1],
    #                         means=[[-2, -4],  # GM1 mean
    #                                ],
    #                         covariances=[[[0.1, 0],  # GM1 mean
    #                                       [0, 0.1]
    #                                       ],
    #                                      ])

    # Define sensor likelihood
    brm = range_model()

    file_ =open('/Users/nick/Downloads/VBIS GM Fusion/nick_output.csv', 'w')
    for i in range(30):
        # Do a VBIS update
        logging.info('Starting VB update...')
        vb = VariationalBayes()
        mu_hat, var_hat, beta_hat = vb.update(measurement, brm, prior)

        # Flatten values
        flat = np.hstack((beta_hat, mu_hat.flatten(), var_hat.flatten()))

        # Save Flattened values
        np.savetxt(file_, np.atleast_2d(flat), delimiter=',')
    file_.close()

def camera_test():
    # Define gridded space for graphing
    min_x, max_x = -5, 5
    min_y, max_y = -5, 5
    res = 100
    x_space, y_space = np.mgrid[min_x:max_x:1/res,
                                min_y:max_y:1/res]
    pos = np.empty(x_space.shape + (2,))
    pos[:, :, 0] = x_space; pos[:, :, 1] = y_space;

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111)

    levels_res = 50
    levels = np.linspace(0, 1, levels_res)

    prior = GaussianMixture(weights=[1, 1, 1, 1, 1],
                            means=[[-2, -4],  # GM1 mean
                                   [-1, -2],  # GM2 mean
                                   [0, 0],  # GM3 mean
                                   [1, -2],  # GM4 mean
                                   [2, -4],  # GM5 mean
                                   ],
                            covariances=[[[0.1, 0],  # GM1 mean
                                          [0, 0.1]
                                          ],
                                         [[0.2, 0],  # GM2 mean
                                          [0, 0.2]
                                          ],
                                         [[0.3, 0],  # GM3 mean
                                          [0, 0.3]
                                          ],
                                         [[0.2, 0],  # GM4 mean
                                          [0, 0.2]
                                          ],
                                         [[0.1, 0],  # GM5 mean
                                          [0, 0.1]],
                                         ])

    min_view_dist = 0.3  # [m]
    max_view_dist = 1.0  # [m]
    detection_model = camera_model_2D(min_view_dist, max_view_dist)
    vb = VariationalBayes()

    # Do a VBIS update
    pose = np.array([0,0,90])
    logging.info('Moving to pose {}.'.format(pose))
    mu, sigma, beta = vb.update(measurement='No Detection',
                                likelihood=detection_model,
                                prior=prior,
                                use_LWIS=True
                                )
    posterior = GaussianMixture(weights=beta, means=mu, covariances=sigma)
    posterior_c = ax.contourf(x_space, y_space, posterior.pdf(pos),
                                        levels=levels)
    plt.colorbar(posterior_c)

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=10, suppress=True)

    # comparison_1d()

    # comparison_2d()
    # gmm_sm_test('Near')
    # compare_to_matlab()

    camera_test()
