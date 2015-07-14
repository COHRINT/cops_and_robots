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
from descartes.patch import PolygonPatch
from shapely.geometry import Polygon


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
    max_num_mixands = 20

    def __init__(self, weights=1, means=0, covariances=1, ellipse_color='red'):
        self.weights = np.asarray(weights)
        self.means = np.asarray(means)
        self.covariances = np.asarray(covariances)
        self.ellipse_color = ellipse_color
        self._input_check()

    def pdf(self, x):
        """Probability density function at state x.

        Will return a probability distribution relative to the shape of the
        input and the dimensionality of the normal. For example, if x is 5x2 
        with a 2-dimensional normal, pdf is 5x1; if x is 5x5x2 
        with a 2-dimensional normal, pdf is 5x5.
        """

        # Ensure proper output shape
        x = np.atleast_1d(x)
        if self.ndims == x.shape[-1]:
            shape = x.shape[:-1]
        else:
            shape = x.shape

        pdf = np.zeros(shape)
        for i, weight in enumerate(self.weights):
            mean = self.means[i]
            covariance = self.covariances[i]
            gaussian_pdf = multivariate_normal.pdf(x, mean, covariance)
            # logging.info(x)
            # logging.info(x.size)
            # logging.info(mean)
            # logging.info(covariance)
            # logging.info(weight)
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

    def max_point_by_grid(self, bounds=[-5,-5,5,5], grid_spacing=0.1):
        #<>TODO: set for n-dimensional
        xx, yy = np.mgrid[bounds[0]:bounds[2]:grid_spacing,
                          bounds[1]:bounds[3]:grid_spacing]
        pos = np.empty(xx.shape + (2,))
        pos[:, :, 0] = xx
        pos[:, :, 1] = yy

        prob = self.pdf(pos)
        MAP_i = np.unravel_index(prob.argmax(), prob.shape)
        MAP_point = np.array([xx[MAP_i[0]][0], yy[0][MAP_i[1]]])
        MAP_prob = prob[MAP_i]
        return MAP_point, MAP_prob

    def std_ellipses(self, num_std=2, resolution=50):
        """
        Generates `num_std` sigma error ellipses for each mixand.

        Note
        ----
        Only applies to two-dimensional Gaussian mixtures.

        Parameters
        ----------
            num_std : The ellipse size in number of standard deviations.
                Defaults to 2 standard deviations.

        Returns
        -------
            A list of Shapely ellipses.

        References
        ----------
        http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
        http://stackoverflow.com/questions/15445546/finding-intersection-points-of-two-ellipses-python
        """
        if self.ndims != 2:
            raise ValueError("Only works for 2-dimensional Gaussian mixtures.")

        def eigsorted(cov):
            """Get 
            """
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]

        ellipses = []

        # Find discrete sin/cos of t
        t = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        st = np.sin(t)
        ct = np.cos(t)

        # Generate all ellipses
        for i, mean in enumerate(self.means):

            # Use eigenvals/vects to get major/minor axes 
            eigvals, eigvects = eigsorted(self.covariances[i])
            a, b = 2 * num_std * np.sqrt(eigvals)

            # Find discrete sin/cos of theta
            theta = np.arctan2(*eigvects[:,0][::-1])
            sth = np.sin(theta)
            cth = np.cos(theta)

            # Find all ellipse points and turn into a Shapely Polygon
            ellipse_pts = np.empty((resolution, 2))
            x0, y0 = mean[0], mean[1]
            ellipse_pts[:, 0] = x0 + a * cth * ct - b * sth * st
            ellipse_pts[:, 1] = y0 + a * sth * ct + b * cth * st
            ellipse = Polygon(ellipse_pts)

            ellipses.append(ellipse)
        return ellipses

    def entropy(self):
        """
        """
        # <>TODO: figure this out. Look at papers!
        # http://www-personal.acfr.usyd.edu.au/tbailey/papers/mfi08_huber.pdf
        pass

    def _merge(self, max_num_mixands=None):
        """
        """
        # <>TODO
        if max_num_mixands:
            self.max_num_mixands = max_num_mixands

        num_mixands = self.weights.size
        if num_mixands <= max_num_mixands:
            logging.info('No need to merge {} mixands.'.format(num_mixands))
            return
        else:
            logging.info('Merging {} mixands down to {}.'.format(num_mixands,
                         self.max_num_mixands))

        # Create dissimilarity matrix B
        B = np.zeros((self.num_mixands, self.num_mixands))
        for i in range(self.num_mixands):
            for j in range(i):
                if i == j:
                    continue
                w_ij = self.weights[i] + self.weights[j]
                w_i_ij = self.weights[i] / (self.weights[i] + self.weights[j])
                w_j_ij = self.weights[j] / (self.weights[i] + self.weights[j])
                mu_ij = w_i_ij * self.means[i] + w_j_ij * self.means[j]

                outer_term = np.outer(self.means[i] - self.means[j],
                                      self.means[i] - self.means[j])
                P_ij = w_i_ij * self.covariances[i] + w_j_ij * self.covariances[j] \
                    + w_i_ij * w_j_ij * outer_term

        # Keep merging until we get the right number of mixands
        while num_mixands > max_num_mixands:
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

    # 2D Gaussian probs
    states = [[0,0],[1,1],[3,3],[4,4]]
    print gauss_2d.pdf(states)

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


def ellipses_test(num_std=2):
    # 2D Gaussian
    gauss_2d = GaussianMixture(weights=[32,
                                        14,
                                        15,
                                        14,
                                        14,
                                        14],
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

    ellipses = gauss_2d.std_ellipses(num_std)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, ellipse in enumerate(ellipses):
        patch = PolygonPatch(ellipse, facecolor=gauss_2d.ellipse_color,
                             alpha=gauss_2d.weights[i])
        ax.add_patch(patch)

    ax.set_xlim([-15, 5])
    ax.set_ylim([-5, 5])
    plt.show()


def fleming_prior():
    fig = plt.figure()

    # Setup spaces
    xx, yy = np.mgrid[-12.5:2.5:1 / 100,
                      -3.5:3.5:1 / 100]
    pos = np.empty(xx.shape + (2,))
    pos[:, :, 0] = xx
    pos[:, :, 1] = yy

    # 2D Gaussian
    gauss_2d = GaussianMixture(weights=[32,
                                        14,
                                        15,
                                        14,
                                        14,
                                        14],
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

    ax = fig.add_subplot(111)
    levels = np.linspace(0, np.max(gauss_2d.pdf(pos)), 50)
    ax.contourf(xx, yy, gauss_2d.pdf(pos), levels=levels, cmap=plt.get_cmap('jet'))
    ax.set_title('2D Gaussian PDF')
    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # pdf_test()
    # rv_test()
    fleming_prior()
    # ellipses_test(2)

