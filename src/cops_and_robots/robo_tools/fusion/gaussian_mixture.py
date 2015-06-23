#!/usr/bin/env python
from __future__ import division
"""Extends multivariate normal to a mixture of multivariate normals.



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
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal, norm


# <>TODO: test for greater than 2D mixtures
class GaussianMixture(object):
    """short description of GaussianMixture

    long description of GaussianMixture

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

    def __init__(self, weights=1, means=0, covariances=1):
        self.weights = np.asarray(weights)
        self.means = np.asarray(means)
        self.covariances = np.asarray(covariances)
        self._input_check()

    def pdf(self, x):
        """Probability density function at state x.
        """

        if x.ndim > 0 and x.size != self.ndims:
            shape = []
            for i in range(self.ndims):
                shape.append(x.shape[0])
        else:
            shape = 1

        pdf = np.zeros(shape)
        for i, weight in enumerate(self.weights):
            mean = self.means[i]
            covariance = self.covariances[i]
            gaussian_pdf = multivariate_normal.pdf(x, mean, covariance)
            pdf += weight * gaussian_pdf

        return pdf

    def rvs(self, size=1):
        """
        """
        c_weights = self.weights.cumsum()  # Cumulative weights
        c_weights = np.hstack([0, c_weights])
        r_weights = np.random.rand(size)  # Randomly sampled weights
        r_weights = np.sort(r_weights)

        if self.ndims > 1:
            rvs = np.zeros((size, self.ndims))
        else:
            rvs = np.zeros(size)
        prev_max = 0
        for i, c_weight in enumerate(c_weights):
            if i == c_weights.size - 1:
                break

            size_i = r_weights[r_weights > c_weight].size
            size_i = size_i - r_weights[r_weights > c_weights[i + 1]].size
            range_ = np.arange(size_i) + prev_max
            range_ = range_.astype(int)

            prev_max = range_[-1]
            mean = self.means[i]
            covariance = self.covariances[i]

            rvs[range_] = multivariate_normal.rvs(mean, covariance, size_i)

        return rvs

    def entropy(self):
        """
        """
        # <>TODO: figure this out. Look at papers!
        # http://www-personal.acfr.usyd.edu.au/tbailey/papers/mfi08_huber.pdf
        pass

    def _input_check(self):
        # Check if weights sum are normalized
        try:
            new_weights = self.weights / np.sum(self.weights)
            assert np.array_equal(self.weights, new_weights)
        except AssertionError, e:
            self.weights = new_weights
            logging.debug("Weights renormalized to {}".format(self.weights))

        # Check if weights sum to 1
        try:
            a = np.sum(self.weights)
            assert np.isclose(np.ones(1), a)
        except AssertionError, e:
            logging.exception('Weights sum to {}, not 1.'.format(a))
            raise e

        # Identify dimensionality
        if self.means.ndim == 0:  # single Univariate gaussian
            self.ndims = 1
            self.weights = np.array([self.weights])
        elif self.means.ndim == 1 and self.weights.size == 1:
            # single multivariate gaussian
            self.ndims = self.means.shape[0]
            self.weights = np.array([self.weights])
        elif self.means.ndim == 1:  # multiple univariate gaussians
            self.ndims = 1
        elif self.means.ndim == 2:  # multiple multivariate gaussians
            self.ndims = self.means.shape[1]

        # Properly format means
        try:
            self.means = self.means.reshape(self.weights.size, self.ndims)
        except ValueError, e:
            logging.exception('Means and weights don\'t agree.')
            raise e

        # Properly format covariances
        try:
            self.covariances = self.covariances.reshape(self.weights.size,
                                                        self.ndims, self.ndims)
        except ValueError, e:
            logging.exception('Covariances and weights don\'t agree.')
            raise e

        # Check if means correspond to variance dimensions
        for i, mean in enumerate(self.means):
            var = self.covariances[i]
            try:
                assert mean.size == var.shape[0]
                assert mean.size == var.shape[1]
            except AssertionError, e:
                logging.exception('Mean {} doesn\'t correspond to variance:'
                                  ' \n{}'.format(mean, var))
                raise e

        # Check if covariances are positive semidefinite
        for var in self.covariances:
            try:
                assert np.all(np.linalg.eigvals(var) >= 0)
            except AssertionError, e:
                logging.exception('Following variance is not positive '
                                  'semidefinite: \n{}'.format(var))
                raise e

        # Check if covariances are symmetric
        for var in self.covariances:
            try:
                tol = 10 ** -6
                a = np.ones_like(var) * tol
                assert np.less(var.T - var, a).all()
            except AssertionError, e:
                logging.exception('Following variance is not symmetric: \n{} '
                                  .format(var))
                raise e


def pdf_test():
    fig = plt.figure()

    # Setup spaces
    x = np.linspace(-5, 5, 100)
    xx, yy = np.mgrid[-5:5:1 / 100,
                      -5:5:1 / 100]
    pos = np.empty(xx.shape + (2,))
    pos[:, :, 0] = xx
    pos[:, :, 1] = yy

    # 1D Gaussian
    gauss_1d = GaussianMixture(1, 3, 0.4)
    ax = fig.add_subplot(221)
    ax.plot(x, gauss_1d.pdf(x), lw=2)
    ax.fill_between(x, 0, gauss_1d.pdf(x), alpha=0.2)
    ax.set_title('1D Gaussian PDF')

    # 2D Gaussian
    gauss_2d = GaussianMixture(weights=1,
                               means=[3, 2],
                               covariances=[[[0.4, 0.3],
                                             [0.3, 0.4]
                                             ],
                                            ])
    ax = fig.add_subplot(222)
    levels = np.linspace(0, np.max(gauss_2d.pdf(pos)), 50)
    ax.contourf(xx, yy, gauss_2d.pdf(pos), levels=levels, cmap=plt.get_cmap('jet'))
    ax.set_title('2D Gaussian PDF')

    # 1D Gaussian Mixutre
    gm_1d = GaussianMixture(weights=[1, 4, 5],
                            means=[3, -3, 1],
                            covariances=[0.4, 0.3, 0.5],
                            )
    ax = fig.add_subplot(223)
    ax.plot(x, gm_1d.pdf(x), lw=2)
    ax.fill_between(x, 0, gm_1d.pdf(x), alpha=0.2,)
    ax.set_title('1D Gaussian Mixture PDF')

    # 2D Gaussian Mixutre
    gm_2d = GaussianMixture(weights=[1, 4, 5],
                            means=[[3, 2],  # GM1 mean
                                   [-3, 4],  # GM2 mean
                                   [1, -1],  # GM3 mean
                                   ],
                            covariances=[[[0.4, 0.3],  # GM1 mean
                                          [0.3, 0.4]
                                          ],
                                         [[0.3, 0.1],  # GM2 mean
                                          [0.1, 0.3]
                                          ],
                                         [[0.5, 0.4],  # GM3 mean
                                          [0.4, 0.5]],
                                         ])
    ax = fig.add_subplot(224)
    levels = np.linspace(0, np.max(gm_2d.pdf(pos)), 50)
    ax.contourf(xx, yy, gm_2d.pdf(pos), levels=levels, cmap=plt.get_cmap('jet'))
    ax.set_title('2D Gaussian Mixture PDF')

    plt.tight_layout()
    plt.show()


def rv_test():
    fig = plt.figure()
    samps_1d = 10000
    samps_2d = 1000000

    # 1D Gaussian
    gauss_1d = GaussianMixture(1, 3, 0.4)
    rvs = gauss_1d.rvs(samps_1d)
    ax = fig.add_subplot(221)
    ax.hist(rvs, histtype='stepfilled', normed=True, alpha=0.2, bins=100)
    ax.set_title('1D Gaussian Samples')

    # 2D Gaussian
    gauss_2d = GaussianMixture(weights=1,
                               means=[3, 2],
                               covariances=[[[0.4, 0.3],
                                             [0.3, 0.4]
                                             ],
                                            ])
    ax = fig.add_subplot(222)
    rvs = gauss_2d.rvs(samps_2d)
    ax.hist2d(rvs[:, 0], rvs[:, 1], bins=50)
    ax.set_title('2D Gaussian Samples')

    # 1D Gaussian Mixutre
    gm_1d = GaussianMixture(weights=[1, 4, 5],
                            means=[3, -3, 1],
                            covariances=[0.4, 0.3, 0.5],
                            )
    rvs = gm_1d.rvs(samps_1d)
    ax = fig.add_subplot(223)
    ax.hist(rvs, histtype='stepfilled', normed=True, alpha=0.2, bins=100)
    ax.set_title('1D Gaussian Mixture Samples')

    # 2D Gaussian Mixutre
    gm_2d = GaussianMixture(weights=[1, 4, 5],
                            means=[[3, 2],  # GM1 mean
                                   [-3, 4],  # GM2 mean
                                   [1, -1],  # GM3 mean
                                   ],
                            covariances=[[[0.4, 0.3],  # GM1 mean
                                          [0.3, 0.4]
                                          ],
                                         [[0.3, 0.1],  # GM2 mean
                                          [0.1, 0.3]
                                          ],
                                         [[0.5, 0.4],  # GM3 mean
                                          [0.4, 0.5]],
                                         ])
    ax = fig.add_subplot(224)
    rvs = gm_2d.rvs(samps_2d)
    ax.hist2d(rvs[:, 0], rvs[:, 1], bins=50)
    ax.set_title('2D Gaussian Mixture Samples')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    pdf_test()
    # rv_test()

