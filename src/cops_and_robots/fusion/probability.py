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
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Probability(object):
    """Abstract base class for probability representation (grid, particle, etc)

    long description of Probability
    
    Parameters
    ----------
    bounds : Array-like
        Bounding coordinates for the probability map.
    res : float
        Resolution used for discretization of the probability map.

    """

    def __init__(self, bounds, res):
        self.bounds = bounds
        self.ndims = int(len(bounds) / 2)
        self.res = res

    def entropy(self):
        """
        """
        # <>TODO: figure this out. Look at papers!
        # http://www-personal.acfr.usyd.edu.au/tbailey/papers/mfi08_huber.pdf
        if not hasattr(self, 'pos'):
            self._discretize()

        if not hasattr(self, 'prob'):
            self.pdf()

        p_i = self.prob #TODO: change to 4 dims.
        H = -np.nansum(p_i * np.log(p_i)) * self.res ** self.ndims # sum of elementwise entropy values
        return H

    def compute_kld(self, other_gm):
        """Computes the KLD of self from another GM.

        Use a truth GM as other_gm.
        """

        q_i = self.prob
        p_i = other_gm.prob

        kld = np.nansum(p_i * np.log(p_i / q_i)) * self.res ** self.ndims
        return kld


    # def _discretize(self, bounds=None, res=None, all_dims=False):
    #     if res is not None:
    #         self.res = res

    #     if bounds is None and self.bounds is None:
    #         b = [-10, 10]  # bounds in any dimension
    #         bounds = [[d] * self.ndims for d in b]  # apply bounds to each dim
    #         self.bounds = [d for dim in bounds for d in dim]  # flatten bounds
    #     elif self.bounds is None:
    #         self.bounds = bounds

    #     # Create grid
    #     if self.ndims == 1:
    #         x = np.arange(self.bounds[0], self.bounds[1], res)
    #         self.x = x
    #         self.pos = x
    #     elif self.ndims == 2:
    #         X, Y = np.mgrid[self.bounds[0]:self.bounds[2] + self.res:self.res,
    #                         self.bounds[1]:self.bounds[3] + self.res:self.res]
    #         pos = np.empty(X.shape + (2,))
    #         pos[:, :, 0] = X; pos[:, :, 1] = Y
    #         self.X = X; self.Y = Y
    #         self.pos = pos

    #     elif self.ndims > 2:

    #         logging.debug('Using first two variables as x and y')
    #         X, Y = np.mgrid[self.bounds[0]:self.bounds[2]
    #                           + res:res,
    #                           self.bounds[1]:self.bounds[3]
    #                           + res:res]
    #         pos = np.empty(X.shape + (2,))
    #         pos[:, :, 0] = X; pos[:, :, 1] = Y
    #         self.X = X; self.Y = Y
    #         self.pos = pos

    #         if all_dims:
    #             #<>TODO: use more than the ndims == 4 case
    #             full_bounds = self.bounds[0:2] + [-0.5, -0.5] \
    #                 + self.bounds[2:] + [0.5, 0.5]
    #             v_spacing = 0.1
    #             grid = np.mgrid[full_bounds[0]:full_bounds[4] + res:res,
    #                             full_bounds[1]:full_bounds[5] + res:res,
    #                             full_bounds[2]:full_bounds[6] + v_spacing:v_spacing,
    #                             full_bounds[3]:full_bounds[7] + v_spacing:v_spacing,
    #                             ]
    #             pos = np.empty(grid[0].shape + (4,))
    #             pos[:, :, :, :, 0] = grid[0]
    #             pos[:, :, :, :, 1] = grid[1]
    #             pos[:, :, :, :, 2] = grid[2]
    #             pos[:, :, :, :, 3] = grid[3]

    #             self.pos_all = pos
    #     else:
    #         logging.error('This should be impossible, a gauss mixture with no variables')
    #         raise ValueError

    def plot(self, title=None, alpha=1.0, show_colorbar=True, **kwargs):
        if not hasattr(self,'ax') or 'ax' in kwargs:
            self.plot_setup(**kwargs)

        if title is None:
            title = self.__str__()

        self.contourf = self.ax.contourf(self.X, self.Y,
                                         self.prob,
                                         levels=self.levels,
                                         # cmap=plt.get_cmap('jet'),
                                         alpha=alpha,
                                         interpolation='none',
                                         antialiased=False
                                         )
        if show_colorbar and not hasattr(self, 'cbar'):
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(self.contourf, cax)
            cbar.ax.tick_params(labelsize=20) 
            self.cbar = cbar
        self.ax.set_title(title, fontsize=20)

        if self.show_ellipses:
            if hasattr(self.distribution, 'camera_viewcone'):
                poly = self.distribution.camera_viewcone
            else:
                poly = None
            self.ellipse_patches = distribution.plot_ellipses(ax=self.ax,
                                                              poly=poly)
        return self.contourf

    def plot_setup(self, fig=None, ax=None, bounds=None, levels=None, 
                   num_levels=50, resolution=0.1, show_ellipses=False):
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
            _, max_prob = self.find_MAP()
            self.levels = np.linspace(0, max_prob * 1.2, num_levels)
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


    def update_plot(self, i=0, **kwargs):
        logging.debug('Probability update {}'.format(i))

        self.plot_remove()
        self.plot(**kwargs)

    def copy(self):
        return deepcopy(self)