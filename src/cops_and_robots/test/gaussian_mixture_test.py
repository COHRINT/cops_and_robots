from __future__ import division

import logging
import pytest
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from scipy.io import loadmat, savemat

from cops_and_robots.fusion.gaussian_mixture import (GaussianMixture,
                                                     generate_random_params)


class TestGaussianMixture:
    diff_tolerance = 10 ** -10

    def test_merging(self, num_mixands=100, max_num_mixands=10, spread=4,
                     speak=False):
        if max_num_mixands is None:
            animate = True
            max_num_mixands = num_mixands
        else:
            animate = False

        # Generate the unmerged and merged gaussians
        weights, means, covariances = generate_random_params(num_mixands,
                                                             ndims=2,
                                                             spread=spread)

        unmerged_gauss_2d = GaussianMixture(weights.copy(),
                                            means.copy(),
                                            covariances.copy(),
                                            max_num_mixands=len(weights))
        merged_gauss_2d = GaussianMixture(weights.copy(),
                                          means.copy(),
                                          covariances.copy(),
                                          max_num_mixands=max_num_mixands)
        matlab_merged_gauss_2d = self.matlab_gm(weights.copy(),
                                                means.copy(),
                                                covariances.copy(),
                                                max_num_mixands)
        mixtures = {'unmerged': unmerged_gauss_2d,
                    'merged': merged_gauss_2d,
                    'matlab merged': matlab_merged_gauss_2d
                    }

        # Setup figure and levels
        fig = plt.figure(figsize=(18,6))
        axes = []
        _, max_1 = unmerged_gauss_2d.max_point_by_grid()
        _, max_2 = merged_gauss_2d.max_point_by_grid()
        _, max_3 = matlab_merged_gauss_2d.max_point_by_grid()
        max_prob = np.max((max_1, max_2, max_3))
        levels = np.linspace(0, max_prob * 1.2, 50)

        # Plot all three
        ax = fig.add_subplot(131)
        axes.append(ax)
        title = 'Original GM ({} mixands)'\
                .format(unmerged_gauss_2d.weights.size)
        unmerged_gauss_2d.plot(ax=ax, levels=levels, title=title)

        ax = fig.add_subplot(132)
        axes.append(ax)
        title = 'Python Merged GM ({} mixands)'\
                .format(merged_gauss_2d.weights.size)
        merged_gauss_2d.plot(ax=ax, levels=levels, title=title)

        ax = fig.add_subplot(133)
        axes.append(ax)
        title = 'Matlab Merged GM ({} mixands)'\
                .format(matlab_merged_gauss_2d.weights.size)
        matlab_merged_gauss_2d.plot(ax=ax, levels=levels, title=title)

        # Add a colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.875, 0.1, 0.025, 0.8])
        fig.colorbar(unmerged_gauss_2d.contourf, cax=cbar_ax)

        class GMAnimation(object):
            """docstring for merged_gm"""
            def __init__(self, parent, mixand_rate=2, levels=None, axes=None):
                self.max_num_mixands = mixtures['unmerged'].weights.size
                self.num_mixands = 1
                self.mixand_rate = mixand_rate
                self.levels = levels
                self.axes = axes
                self.parent = parent

            def update(self,i=0):
                # Regenerate GMs
                merged_gauss_2d = GaussianMixture(weights, means, covariances,
                                                  max_num_mixands=self.num_mixands)
                matlab_merged_gauss_2d = self.parent.matlab_gm(weights,
                                                   means,
                                                   covariances,
                                                   max_num_mixands=self.num_mixands)

                # Replot GMs
                title = 'Python Merged GM ({} mixands)'.format(merged_gauss_2d.weights.size)
                if hasattr(self,'old_contour'):
                    merged_gauss_2d.contourf = self.old_contour
                    merged_gauss_2d.plot_remove()
                self.old_contour = merged_gauss_2d.plot(ax=self.axes[1], levels=self.levels, title=title)

                title = 'Matlab Merged GM ({} mixands)'.format(matlab_merged_gauss_2d.weights.size)
                if hasattr(self,'old_matlab_contour'):
                    matlab_merged_gauss_2d.contourf = self.old_matlab_contour
                    matlab_merged_gauss_2d.plot_remove()
                self.old_matlab_contour = matlab_merged_gauss_2d.plot(ax=self.axes[2], levels=self.levels, title=title)

                # Decrement mixands (with wrapping)
                if self.num_mixands == self.max_num_mixands:
                    self.num_mixands = 1
                elif np.int(self.num_mixands * self.mixand_rate) < self.max_num_mixands:
                    self.num_mixands = np.int(self.num_mixands * self.mixand_rate)
                else:
                    self.num_mixands = self.max_num_mixands

            def compare_results():
                pass

        if animate:
            gm_ani = GMAnimation(self, mixand_rate=2, levels=levels, axes=axes)
            ani = animation.FuncAnimation(fig, gm_ani.update, 
                interval=100,
                repeat=True,
                blit=False,
                )
        else:
            self.diff(merged_gauss_2d, matlab_merged_gauss_2d)
        plt.show()
        self.check_diff()

    def diff(self, python_gm, matlab_gm):
        # Sort everything before comparing
        py_inds = python_gm.weights.argsort()
        mat_inds = matlab_gm.weights.argsort()
        python_gm.weights = python_gm.weights[py_inds]
        matlab_gm.weights = matlab_gm.weights[mat_inds]
        python_gm.means = python_gm.means[py_inds]
        matlab_gm.means = matlab_gm.means[mat_inds]
        python_gm.covariances = python_gm.covariances[py_inds]
        matlab_gm.covariances = matlab_gm.covariances[mat_inds]

        diff_weights = python_gm.weights - matlab_gm.weights
        diff_means = (python_gm.means - matlab_gm.means).flatten()
        diff_covariances = (python_gm.covariances - matlab_gm.covariances).flatten()

        fig = plt.figure(figsize=(18,6))
        ax = fig.add_subplot(131)
        x = np.arange(diff_weights.size)
        ax.bar(x, diff_weights, align='center')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_title('Difference in weights')

        ax = fig.add_subplot(132)
        x = np.arange(diff_means.size)
        ax.bar(x, diff_means, align='center')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_title('Difference in means')

        ax = fig.add_subplot(133)
        x = np.arange(diff_covariances.size)
        ax.bar(x, diff_covariances, align='center')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_title('Difference in variances')

        self.diffs = np.hstack((diff_weights, diff_means, diff_covariances))

    def matlab_gm(self, weights, means, covariances, max_num_mixands):
        mdict = {'weights': weights,
                 'means': means,
                 'covariances': covariances,
                 'max_num_mixands': max_num_mixands,
                 }

        savemat('matlab/gaussian_mixture/data/from_python.mat', mdict)
        raw_input("Hit enter when Matlab has created some output to use...")
        mdict = loadmat('matlab/gaussian_mixture/data/from_matlab.mat')

        # Get weights, means, covariances from MATLAB
        weights = mdict['weights'][0]
        means = mdict['means']
        covariances = mdict['covariances'].flatten('F').reshape(-1,2,2)
        gm = GaussianMixture(weights, means, covariances)
        return gm

    def check_diff(self):
        assert (self.diffs < self.diff_tolerance).all()
