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

from cops_and_robots.robo_tools.fusion.softmax import (speed_model,
                                                       intrinsic_space_model)
from cops_and_robots.robo_tools.fusion.binary_softmax import binary_speed_model
from cops_and_robots.robo_tools.fusion.gaussian_mixture import GaussianMixture


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

    def __init__(self, num_convergence_loops=15,
                 tolerance=10 ** -3,
                 max_EM_steps=1000):
        self.num_convergence_loops = num_convergence_loops
        self.tolerance = tolerance
        self.max_EM_steps = max_EM_steps

    def vb_update(self, measurement, likelihood, prior,
                  init_mean=0, init_var=1, init_alpha=0.5, init_xi=1):
        """Variational bayes update for Gaussian and Softmax.
        """

        xis, alpha, mu_hat, var_hat, prior_mean, prior_var = \
            self._check_inputs(init_mean, init_var, init_alpha, init_xi, prior)

        # Likelihood values
        m = len(likelihood.class_labels)
        w = likelihood.weights
        b = likelihood.biases
        j = likelihood.class_labels.index(measurement)

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
            for i in range(self.num_convergence_loops):

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
            log_c_hat = y_cs[j] - alpha + sum1 - KLD

            if np.abs(log_c_hat - prev_log_c_hat) < self.tolerance:
                logging.info('Convergence reached at step {}!'.format(EM_step))
                break

            prev_log_c_hat = log_c_hat
            EM_step += 1

        if mu_hat.size == 1:
            mu_post = mu_hat[0]
        else:
            mu_post = mu_hat
        if var_hat.size == 1:
            var_post = var_hat[0][0]
        else:
            var_post = var_hat

        logging.info('VB update found mean of {} and variance of {}.'
                     .format(mu_post, var_post))

        return mu_post, var_post, log_c_hat

    def vbis_update(self, measurement, likelihood, prior,
                    init_mean=0, init_var=1, init_alpha=0.5, init_xi=1,
                    num_samples=1500):
        """VB update with importance sampling for Gaussian and Softmax.
        """
        mu_VB, var_VB, log_c_hat = self.vb_update(measurement, likelihood,
                                                  prior,
                                                  init_mean, init_var,
                                                  init_alpha, init_xi)

        #<>EXTEND
        prior_var = np.asarray(prior.covariances[0])

        # <>TODO: include GMM posteriors as well
        # Importance distribution
        q = GaussianMixture(1, mu_VB, prior_var)

        # Importance sampling correction
        w = np.zeros(num_samples)  # Importance weights
        x = q.rvs(size=num_samples)  # Sampled points
        x = np.asarray(x)
        for i in range(num_samples):
            w[i] = prior.pdf(x[i]) \
                * likelihood.probs_at_state(x[i], measurement)\
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
            mu_post_vbis = mu_hat[0]
        else:
            mu_post_vbis = mu_hat
        if var_hat.size == 1:
            var_post_vbis = var_hat[0][0]
        else:
            var_post_vbis = var_hat

        logging.info('VBIS update found mean of {} and variance of {}.'
                     .format(mu_post_vbis, var_post_vbis))

        return mu_post_vbis, var_post_vbis, log_c_hat

    def vbis_update_mms_gmm(self, measurement, likelihood, prior,
                    num_vbis_samples=1500, num_mixand_samples=1500):
        """VB update using Gaussian mixtures and multimodal softmax.
        """

        h = 0
        K = len(likelihood.subclass_labels) * prior.weights.size
        mu_hat = np.zeros((K, prior.means.shape[0]))
        var_hat = np.zeros((K, prior.covariances.shape[0],
                            prior.covariances.shape[1]))
        beta_hat = np.zeros(K)

        for r, subclass in enumerate(likelihood.subclasses): # <>FIX
            for u, gm_weight in enumerate(prior.weights):

                # Compute \hat{P}_s(r|u)
                mixand = gaussian_mixture(1, prior.means[u],
                                          prior.covariances[u])
                mixand_samples = mixand.rvs(num_mixand_samples)
                p_hat_ru_sampled = 0
                for mixand_sample in mixand_samples:
                    p_hat_ru_sampled += subclass.probs_at_state(mixand_sample,r)

                p_hat_ru_sampled = p_hat_ru_sampled / num_mixand_samples

                # Find parameters via VBIS fusion
                sm_likelihood = Softmax() #<>FIX
                subclass_measurement = subclass.class_labels[r]

                mu_hat[h, :], var_hat[h,:], log_c_hat = \
                    self.vbis_update(subclass_measurement, subclass_likelihood,
                                     mixand)

                # Compute \hat{P}(r|u)
                p_hat_ru = np.max(np.exp(log_c_hat), p_hat_ru_sampled)

                # Find P(u,r|D_k) \approxequal \hat{B}_{ur}
                beta_hat[h] = gm_weight * p_hat_ru

                h += 1

        # Renormalize \hat{B}_{ur}
        beta_hat /= np.sum(beta_hat)

        return mu_hat, var_hat, beta_hat

    def _lambda(self, xi_c):
        return 1 / (2 * xi_c) * ( (1 / (1 + np.exp(-xi_c))) - 0.5)

    def _check_inputs(self, init_mean, init_var, init_alpha, init_xi, prior):
        # Make sure inputs are numpy arrays
        init_mean = np.asarray(init_mean)
        init_var = np.asarray(init_var)
        init_alpha = np.asarray(init_alpha)
        init_xi = np.asarray(init_xi)

        if init_xi.ndim != 1:
            try:
                m = len(likelihood.class_labels)
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
    logging.info('Nisar\'s VB update had mean difference {} and var difference {}'
                 .format(diff_vb_mean, diff_vb_var))

    # Do a VBIS update
    vbis_mean, vbis_var, _ = vb.vbis_update(measurement, sm, prior, init_mean,
                                         init_var, init_alpha, init_xi)
    vbis_posterior = GaussianMixture(1, vbis_mean, vbis_var)

    nisar_vbis_mean = 0.154223416817080
    nisar_vbis_var = 0.00346064073274943
    diff_vbis_mean = vbis_mean - nisar_vbis_mean
    diff_vbis_var = vbis_var - nisar_vbis_var
    logging.info('Nisar\'s VBIS update had mean difference {} and var difference {}'
                 .format(diff_vbis_mean, diff_vbis_var))

    # Plot results
    likelihood_label = 'Likelihood of \'{}\''.format(measurement)
    ax = sm.plot_class(measurement_i, fill_between=False, label=likelihood_label, ls='--')
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
    measurement_i = sm.class_labels.index(measurement)

    # Do a VB update
    init_mean = np.zeros((1,2))
    init_var = np.eye(2)
    init_alpha = 0.5
    init_xi = np.ones(5)

    vb = VariationalBayes()
    vb_mean, vb_var, _ = vb.vb_update(measurement, sm, prior, init_mean,
                                       init_var, init_alpha, init_xi)
    # vb_mean, vb_var = vb.vbis_update(measurement, sm, prior, init_mean,
    #                                    init_var, init_alpha, init_xi)
    vb_posterior = GaussianMixture(1, vb_mean, vb_var)

    nisar_vb_mean = np.array([1.795546121012238, 2.512627005425541])
    nisar_vb_var = np.array([[0.755723395661314, 0.091742424424428],
                             [0.091742424424428, 0.747611340151417]])
    diff_vb_mean = vb_mean - nisar_vb_mean
    diff_vb_var = vb_var - nisar_vb_var
    logging.info('Nisar\'s VB update had mean difference: \n {}\n and var difference: \n {}'
                 .format(diff_vb_mean, diff_vb_var))

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
    sm.plot_class(measurement_i, ax=lh_ax, label=likelihood_label, ls='--', levels=lh_levels)
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


def gmm_sm_test():

    # Define prior 
    prior = GaussianMixture(weights=[1, 4, 5],
                        means=[3, -3, 1],
                        covariances=[0.4, 0.3, 0.5],
                        )
    min_x, max_x = -5, 5
    res = 10000
    x_space = np.linspace(min_x, max_x, res)

    # Define sensor likelihood
    bsm = binary_speed_model()
    measurement = 'Not Slow'
    measurement_i = sm.class_labels.index(measurement)

    # Do a VBIS update
    vb = VariationalBayes()
    mu_hat, var_hat, beta_hat = vb.vbis_update_mms_gmm(self, measurement, bsm, prior)
    vbis_posterior = GaussianMixture(weights=beta_hat, means=mu_hat, covariances=var_hat)

    # Plot results
    likelihood_label = 'Likelihood of \'{}\''.format(measurement)
    ax = bsm.plot_class(measurement_i, fill_between=False, label=likelihood_label, ls='--')
    ax.plot(x_space, prior.pdf(x_space), lw=1, label='prior pdf', c='grey', ls='--')

    ax.plot(x_space, vbis_posterior.pdf(x_space), lw=2, label='VBIS Posterior', c='g')
    ax.fill_between(x_space, 0, vbis_posterior.pdf(x_space), alpha=0.2, facecolor='g')

    ax.set_title('VBIS Update - Gaussian Mixtures and MMS')
    ax.legend()
    ax.set_xlim([0, 0.4])
    ax.set_ylim([0, 7])
    plt.show()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=10, suppress=True)

    comparison_1d()

    comparison_2d()

    # gmm_sm_test()

