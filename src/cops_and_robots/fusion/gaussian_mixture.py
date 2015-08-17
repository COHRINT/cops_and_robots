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
import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from scipy.stats import multivariate_normal, norm, entropy
from descartes.patch import PolygonPatch
from shapely.geometry import Polygon


# <>TODO: test for greater than 2D mixtures
class GaussianMixture(object):
    """A collection of weighted multivariate normal distributions.

    A Gaussian mixture is a collection of mixands: individual multivariate 
    normals. It takes the form:

    .. math::

        f(\\mathbf{x}) = \\sum_{i=1}^n \\frac{w_i}
            {\\sqrt{(2\\pi)^d \\vert \\mathbf{P}_i \\vert}}
            \\exp{\\left[-\\frac{1}{2}(\\mathbf{x} - \\mathbf{\\mu}_i)^T
            \\mathbf{P}_i^{-1} (\\mathbf{x} - \\mathbf{\\mu}_i) \\right]}

    Where `d` is the dimensionality of the state vector `x`, and each mixand 
    `i` has weight, mean and covariances `w`, `mu` and `P`.

    Parameters
    ----------
    weights : array_like, optional
        Scaling factor for each mixand.

    Attributes
    ----------
    attr : attr_type
        attr_description

    Methods
    ----------
    attr : attr_type
        attr_description

    """

    def __init__(self, weights=1, means=0, covariances=1, ellipse_color='red',
                 max_num_mixands=20):
        self.weights = np.asarray(weights, dtype=np.float)
        self.means = np.asarray(means, dtype=np.float)
        self.covariances = np.asarray(covariances, dtype=np.float)
        self.ellipse_color = ellipse_color
        self.max_num_mixands = max_num_mixands
        self.num_mixands = self.weights.size
        self._input_check()

    def __str__(self):
        d = {}
        for i, weight in enumerate(self.weights):
            d['Mixand {}'.format(i)] = np.hstack((weight,
                                                  self.means[i],
                                                  self.covariances[i].flatten()
                                                  ))
        ind = ['Weight'] + ['Mean'] * self.ndims + ['Variance'] * self.ndims ** 2
        df = pd.DataFrame(d, index=ind)
        return '\n' + df.to_string()
            

    def pdf(self, x=None):
        """Probability density function at state x.

        Will return a probability distribution relative to the shape of the
        input and the dimensionality of the normal. For example, if x is 5x2 
        with a 2-dimensional normal, pdf is 5x1; if x is 5x5x2 
        with a 2-dimensional normal, pdf is 5x5.
        """

        # Look over the whole state space
        if x is None:
            if not hasattr(self, 'pos'):
                self._discretize()
            x = self.pos

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
            gaussian_pdf = multivariate_normal.pdf(x, mean, covariance,
                                                   allow_singular=True)
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

    def max_point_by_grid(self, bounds=None, grid_spacing=0.1):
        #<>TODO: set for n-dimensional
        if not hasattr(self, 'pos'):
            self._discretize(bounds, grid_spacing)

        prob = self.pdf(self.pos)
        MAP_i = np.unravel_index(prob.argmax(), prob.shape)
        MAP_point = np.array([self.xx[MAP_i[0]][0], self.yy[0][MAP_i[1]]])
        MAP_prob = prob[MAP_i]
        return MAP_point, MAP_prob

    def copy(self):
        return deepcopy(self)

    def std_ellipses(self, num_std=1, resolution=20):
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

    def plot_ellipses(self, ax=None, lw=20, poly=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        ellipses = self.std_ellipses(**kwargs)
        ellipse_patches = []
        for i, ellipse in enumerate(ellipses):
            if poly is not None:
                if poly.intersects(ellipse):
                    ec = 'white'
                else:
                    ec = 'black'
            else:
                ec = 'black'
            patch = PolygonPatch(ellipse, facecolor='none', edgecolor=ec,
                                 linewidth=self.weights[i] * lw,
                                 zorder=15)
            ax.add_patch(patch)
            ellipse_patches.append(patch)
        return ellipse_patches

    def plot(self, title=None, alpha=1.0, **kwargs):
        if not hasattr(self,'ax'):
            self.plot_setup(**kwargs)
        if title is None:
            title = 'Gaussian Mixture ({} mixands)'.format(self.num_mixands)
        self.contourf = self.ax.contourf(self.xx, self.yy,
                                         self.pdf(self.pos),
                                         levels=self.levels,
                                         cmap=plt.get_cmap('jet'),
                                         alpha=alpha,
                                         )
        self.ax.set_title(title)

        if self.show_ellipses:
            if hasattr(self.distribution, 'camera_viewcone'):
                poly = self.distribution.camera_viewcone
            else:
                poly = None
            self.ellipse_patches = distribution.plot_ellipses(ax=self.ax,
                                                              poly=poly)
        return self.contourf

    def plot_setup(self, fig=None, ax=None, bounds=None, levels=None,
                   resolution=0.1, show_ellipses=False):
        self.show_ellipses = show_ellipses
        if fig is None:
            self.fig = plt.gcf()
        else:
            self.fig = fig

        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax

        if bounds is None:
            bounds = self.bounds

        if not hasattr(self,'pos'):
            self._discretize(bounds=bounds)

        # Set levels
        if levels is None:
            max_prob = np.max(self.pdf(self.pos))
            self.levels = np.linspace(0, max_prob * 1.2, 50)
        else:
            self.levels = levels
        
        # Set bounds
        plt.axis('scaled')
        self.ax.set_xlim([bounds[0], bounds[2]])
        self.ax.set_ylim([bounds[1], bounds[3]])

    def plot_remove(self):
        """Removes all plotted elements related to this gaussian mixture.
        """
        if hasattr(self,'contourf'):
            for collection in self.contourf.collections:
                collection.remove()
            del self.contourf

        if hasattr(self, 'ellipse_patches'):
            for patch in self.ellipse_patches:
                patch.remove()
            del self.ellipse_patches

    def entropy(self):
        """
        """
        # <>TODO: figure this out. Look at papers!
        # http://www-personal.acfr.usyd.edu.au/tbailey/papers/mfi08_huber.pdf
        if not hasattr(self,'pos'):
            self._discretize()

        p_i = self.pdf(self.pos) 
        # p_i /= p_i.sum()  # normalize input probability
        # H = np.sum(entr(p_i)) * self.grid_spacing ** self.ndims # sum of elementwise entropy values
        H = -np.sum(p_i * np.log(p_i)) * self.grid_spacing ** self.ndims # sum of elementwise entropy values
        return H

    def combine_gms(self, other_gm, self_gm_weight=0.5):
        """Merge two gaussian mixtures together, weighing the original by alpha.
        """
        alpha = self_gm_weight
        beta_hat = np.hstack((self.weights * alpha, other_gm.weights * (1 - alpha)))
        mu_hat = np.vstack((self.means, other_gm.means))
        var_hat = np.concatenate((self.covariances, other_gm.covariances))
        return GaussianMixture(beta_hat, mu_hat, var_hat)

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
            assert np.isclose(np.ones(1), np.sum(self.weights))
        except AssertionError, e:
            logging.exception('Weights sum to {}, not 1.'.format(np.sum(self.weights)))
            raise

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

        # Merge if necessary
        self._merge()

    def _discretize(self, bounds=None, grid_spacing=0.1):
        self.grid_spacing = grid_spacing
        if bounds is None:
            b = [-10, 10]  # bounds in any dimension
            bounds = [[d] * self.ndims for d in b]  # apply bounds to each dim
            self.bounds = [d for dim in bounds for d in dim]  # flatten bounds
        else:
            self.bounds = bounds

        # Create grid
        if self.ndims == 1:
            x = np.arange(self.bounds[0], self.bounds[1], grid_spacing)
            self.x = x
            self.pos = x
        elif self.ndims == 2:
            xx, yy = np.mgrid[self.bounds[0]:self.bounds[2]
                              + grid_spacing:grid_spacing,
                              self.bounds[1]:self.bounds[3]
                              + grid_spacing:grid_spacing]
            pos = np.empty(xx.shape + (2,))
            pos[:, :, 0] = xx; pos[:, :, 1] = yy
            self.xx = xx; self.yy = yy
            self.pos = pos
        else:
            logging.error('Only discretizing 2- or 1-dimensional GMs.')
            raise ValueError

    def _merge(self, max_num_mixands=None):
        """
        """
        if max_num_mixands is None:
            max_num_mixands = self.max_num_mixands

        # Check if merging is useful
        if self.num_mixands <= max_num_mixands:
            logging.debug('No need to merge {} mixands.'
                          .format(self.num_mixands))
            return
        else:
            logging.debug('Merging {} mixands down to {}.'
                          .format(self.num_mixands, self.max_num_mixands))

        # Create lower-triangle of dissimilarity matrix B
        #<>TODO: this is O(n ** 2) and very slow. Speed it up! parallelize?
        B = np.zeros((self.num_mixands, self.num_mixands))
        for i in range(self.num_mixands):
            mix_i = (self.weights[i], self.means[i], self.covariances[i])
            for j in range(i):
                if i == j:
                    continue
                mix_j = (self.weights[j], self.means[j], self.covariances[j])
                B[i,j] = mixand_dissimilarity(mix_i, mix_j)

        # Keep merging until we get the right number of mixands
        deleted_mixands = []
        while self.num_mixands > max_num_mixands:
            # Find most similar mixands
            try:
                #<>TODO: replace with infinities, not 0
                min_B = B[B>0].min()
            except ValueError, e:
                logging.error('Could not find a minimum value in B: \n{}'
                              .format(B))
                raise e
            ind = np.where(B==min_B)
            i, j = ind[0][0], ind[1][0]

            # Get merged mixand
            mix_i = (self.weights[i], self.means[i], self.covariances[i])
            mix_j = (self.weights[j], self.means[j], self.covariances[j])
            w_ij, mu_ij, P_ij = merge_mixands(mix_i, mix_j)

            # Replace mixand i with merged mixand
            ij = i
            self.weights[ij] = w_ij
            self.means[ij] = mu_ij
            self.covariances[ij] = P_ij

            # Fill mixand i's B values with new mixand's B values
            mix_ij = (w_ij, mu_ij, P_ij)
            deleted_mixands.append(j)
            for k in range(B.shape[0]):
                if k == ij or k in deleted_mixands:
                    continue

                # Only fill lower triangle
                mix_k = (self.weights[k], self.means[k], self.covariances[k])
                if k < i:
                    B[ij,k] = mixand_dissimilarity(mix_k, mix_ij)
                else:
                    B[k,ij] = mixand_dissimilarity(mix_k, mix_ij)

            # Remove mixand j from B
            B[j,:] = np.inf
            B[:,j] = np.inf
            self.num_mixands -= 1

        # Delete removed mixands from parameter arrays
        self.weights = np.delete(self.weights, deleted_mixands, axis=0)
        self.means = np.delete(self.means, deleted_mixands, axis=0)
        self.covariances = np.delete(self.covariances, deleted_mixands, axis=0)

def entr(p_i):
    if p_i > 0:
       return -p_i * np.log(p_i)
    elif np.isclose(p_i, 0):
        return 0
    else:
      return -np.inf

def merge_mixands(mix_i, mix_j):
    """Use moment-preserving merge (0th, 1st, 2nd moments) to combine mixands.
    """
    # Unpack mixands
    w_i, mu_i, P_i = mix_i
    w_j, mu_j, P_j = mix_j

    # Merge weights
    w_ij = w_i + w_j
    w_i_ij = w_i / (w_i + w_j)
    w_j_ij = w_j / (w_i + w_j)

    # Merge means
    mu_ij = w_i_ij * mu_i + w_j_ij * mu_j

    # Merge covariances
    P_ij = w_i_ij * P_i + w_j_ij * P_j + \
        w_i_ij * w_j_ij * np.outer(mu_i - mu_j, mu_i - mu_j)

    return w_ij, mu_ij, P_ij


def mixand_dissimilarity(mix_i, mix_j):
    """Calculate KL descriminiation-based dissimilarity between mixands.
    """
    # Get covariance of moment-preserving merge
    w_i, mu_i, P_i = mix_i
    w_j, mu_j, P_j = mix_j
    _, _, P_ij = merge_mixands(mix_i, mix_j)

    # Use slogdet to prevent over/underflow
    _, logdet_P_ij = np.linalg.slogdet(P_ij)
    _, logdet_P_i = np.linalg.slogdet(P_i)
    _, logdet_P_j = np.linalg.slogdet(P_j)
    
    # <>TODO: check to see if anything's happening upstream
    if np.isinf(logdet_P_ij):
        logdet_P_ij = 0
    if np.isinf(logdet_P_i):
        logdet_P_i = 0
    if np.isinf(logdet_P_j):
        logdet_P_j = 0

    b = 0.5 * ((w_i + w_j) * logdet_P_ij - w_i * logdet_P_i - w_j * logdet_P_j)
    return b


def generate_random_params(num_mixands, ndims=2, spread=4):
    # Randomly generate parameters
    weights = np.random.uniform(size=num_mixands)
    means = np.random.randn(num_mixands, 2) * spread
    covariances = np.abs(np.random.randn(num_mixands, 2, 2))
    for i, covariance in enumerate(covariances):
        s = np.sort(covariance, axis=None)
        covariance[0,1] = s[0]
        covariance[1,0] = s[1]
        if int(s[3] * 100000) % 2:
            covariance[0,0] = s[3]
            covariance[1,1] = s[2]
        else:
            covariance[0,0] = s[2]
            covariance[1,1] = s[3]
        covariances[i] = (covariance + covariance.T)/2

    return weights, means, covariances


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
    return GaussianMixture(weights=[32,
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


def uniform_prior(num_mixands=10, bounds=None):
    if bounds is None:
        bounds = [-5, -5, 5, 5]

    n = np.int(np.sqrt(num_mixands))
    num_mixands = n ** 2
    weights = np.ones(num_mixands)
    mu_x = np.linspace(bounds[0], bounds[2], num=n)
    mu_y = np.linspace(bounds[1], bounds[3], num=n)
    mu_xx, mu_yy = np.meshgrid(mu_x, mu_y)
    means = np.dstack((mu_xx, mu_yy)).reshape(-1,2)
    covariances = np.ones((num_mixands,2,2)) * 1000
    for i, cov in enumerate(covariances):
        covariances[i] = cov - np.roll(np.eye(2), 1, axis=0) * 1000

    return GaussianMixture(weights, means, covariances)


def fleming_prior_test():
    fig = plt.figure()

    bounds = [-12.5, -3.5, 2.5, 3.5]
    # Setup spaces
    res = 1/100
    xx, yy = np.mgrid[bounds[0]:bounds[2]:res,
                      bounds[1]:bounds[3]:res,
                      ]
    pos = np.empty(xx.shape + (2,))
    pos[:, :, 0] = xx
    pos[:, :, 1] = yy

    # 2D Gaussian
    gauss_2d = fleming_prior()
    ax = fig.add_subplot(111)
    levels = np.linspace(0, np.max(gauss_2d.pdf(pos)), 50)
    cax = ax.contourf(xx, yy, gauss_2d.pdf(pos), levels=levels, cmap=plt.get_cmap('jet'))
    ax.set_title('2D Gaussian PDF')
    fig.colorbar(cax)

    plt.axis('scaled')
    ax.set_xlim(bounds[0:3:2])
    ax.set_ylim(bounds[1:4:2])
    plt.show()


def uniform_prior_test(num_mixands=10, bounds=None):
    if bounds is None:
        bounds = [-5, -5, 5, 5]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Setup spaces
    res = 1/100
    xx, yy = np.mgrid[bounds[0]:bounds[2]:res,
                      bounds[1]:bounds[3]:res,
                      ]
    pos = np.empty(xx.shape + (2,))
    pos[:, :, 0] = xx
    pos[:, :, 1] = yy

    gauss_2d = uniform_prior()
    levels = np.linspace(0, np.max(gauss_2d.pdf(pos)), 50)
    cax = ax.contourf(xx, yy, gauss_2d.pdf(pos), levels=levels, cmap=plt.get_cmap('jet'))
    ax.set_title('2D Gaussian PDF')
    fig.colorbar(cax)

    plt.axis('scaled')
    ax.set_xlim(bounds[0:3:2])
    ax.set_ylim(bounds[1:4:2])
    plt.show()


def merge_test(num_mixands=100, max_num_mixands=10, spread=4,):
    # Generate the unmerged and merged gaussians
    weights, means, covariances = generate_random_params(num_mixands,
                                                         ndims=2,
                                                         spread=spread)
    merged_gauss_2d = GaussianMixture(weights.copy(),
                                      means.copy(),
                                      covariances.copy(),
                                      max_num_mixands=max_num_mixands)
    

def entropy_test():

    # 1D entropy - no mixtures ################################################
    gauss1_1d = GaussianMixture(1, 3, 1.5)
    gauss2_1d = GaussianMixture(1, 3, 0.5)

    # Discrete
    H1_1d = gauss1_1d.entropy()
    H2_1d = gauss2_1d.entropy()

    # Exact
    var1 = gauss1_1d.covariances[0][0][0]
    var2 = gauss2_1d.covariances[0][0][0]
    exactH1_1d = 0.5 * np.log(var1 * 2 * np.pi * np.e)
    exactH2_1d = 0.5 * np.log(var2 * 2 * np.pi * np.e)

    # Scipy
    x = np.arange(-10, 10, 1)
    scipyH1_1d = entropy(norm(3, np.sqrt(1.5)).pdf(x))
    scipyH2_1d = entropy(norm(3, np.sqrt(0.5)).pdf(x))

    logging.info('Entropy of a 1d gaussian: {} (discrete) \t {} (exact) \t {} (scipy)'
                 '\n with params: {}'
                 .format(H1_1d, exactH1_1d, scipyH1_1d, gauss1_1d))
    logging.info('Entropy of a 1d gaussian: {} (discrete) \t {} (exact) \t {} (scipy)'
                 '\n with params: {}'
                 .format(H2_1d, exactH2_1d, scipyH2_1d, gauss2_1d))

    # 1D entropy - mixtures ###################################################
    gauss3_1d = GaussianMixture([0.3, 0.7], [-3, 6], [1.5, 4])
    gauss4_1d = GaussianMixture([0.3, 0.7], [-3, 6], [0.5, 0.4])

    # Discrete
    H3_1d = gauss3_1d.entropy()
    H4_1d = gauss4_1d.entropy()

    logging.info('Entropy of a 1d gaussian: {} (discrete)'
                 '\n with params: {}'
                 .format(H3_1d, gauss3_1d))
    logging.info('Entropy of a 1d gaussian: {} (discrete)'
                 '\n with params: {}'
                 .format(H4_1d, gauss4_1d))
    #<>TODO: VALIDATE AGAINST NISAR'S CODE

    # 2D entropy - no mixtures ################################################
    gauss1_2d = GaussianMixture(1, [3, -2], [[1.5, 1.0],[1.0, 1.5]])
    gauss2_2d = GaussianMixture(1, [3, -2], [[0.5, 0.0],[0.0, 0.5]])

    # Discrete
    H1_2d = gauss1_2d.entropy()
    H2_2d = gauss2_2d.entropy()

    # Exact
    det_var1 = np.linalg.det(gauss1_2d.covariances)
    det_var2 = np.linalg.det(gauss2_2d.covariances)
    exactH1_2d = 0.5 * np.log(det_var1 * (2 * np.pi * np.e) ** gauss1_2d.ndims)
    exactH2_2d = 0.5 * np.log(det_var2 * (2 * np.pi * np.e) ** gauss2_2d.ndims)

    logging.info('Entropy of a 2d gaussian: {} (discrete) \t {} (exact)'
                 '\n with params: {}'
                 .format(H1_2d, exactH1_2d, gauss1_2d))
    logging.info('Entropy of a 2d gaussian: {} (discrete) \t {} (exact)'
                 '\n with params: {}'
                 .format(H2_2d, exactH2_2d, gauss2_2d))

    # 2D entropy - mixtures ###################################################
    gauss3_2d = GaussianMixture([0.3, 0.7],
                                [[3, -2],
                                 [-4, -6]
                                 ], 
                                 [[[1.5, 1.0],
                                   [1.0, 1.5]],
                                  [[2.5, -0.3],
                                   [-0.3, 2.5]]
                                 ])
    gauss4_2d = GaussianMixture([0.3, 0.7],
                                [[3, -2],
                                 [-4, -6]
                                 ], 
                                 [[[0.5, 0.0],
                                   [0.0, 0.5]],
                                  [[1.5, -0.3],
                                   [-0.3, 1.5]]
                                 ])

    # Discrete
    H3_2d = gauss3_2d.entropy()
    H4_2d = gauss4_2d.entropy()

    logging.info('Entropy of a 2d gaussian: {} (discrete)'
                 '\n with params: {}'
                 .format(H3_2d, gauss3_2d))
    logging.info('Entropy of a 2d gaussian: {} (discrete)'
                 '\n with params: {}'
                 .format(H4_2d, gauss4_2d))

def merge_gm_test(alpha=0.5):
    gm1 = GaussianMixture([0.3, 0.7],
                                [[3, -2],
                                 [-4, -6]
                                 ], 
                                 [[[0.5, 0.0],
                                   [0.0, 0.5]],
                                  [[1.5, -0.3],
                                   [-0.3, 1.5]]
                                 ])

    gm2 = GaussianMixture([0.3, 0.7],
                                [[-3, 2],
                                 [4, 6]
                                 ], 
                                 [[[0.5, 0.0],
                                   [0.0, 0.5]],
                                  [[1.5, -0.3],
                                   [-0.3, 1.5]]
                                 ])

    gm3 = gm1.combine_gms(gm2, alpha)
    print gm3

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # pdf_test()
    # rv_test()
    # fleming_prior_test()
    
    # fp = fleming_prior_test()
    # new_fp = fp.copy()
    # new_fp.weights = np.ones(6)
    # print fp
    # ellipses_test(2)
    # merge_test(120)
    # uniform_prior_test()
    # entropy_test()
    merge_gm_test(0.01)
