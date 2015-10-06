#!/usr/bin/env python
"""Provides a continuous probability representation of robot locations.

When using the *continuous* ``fusion_engine`` type, the probability
layer is used to represent the distribution of a cop's expecation of
robber locations. One probability layer exists per robot, as well as
one additional probability layer to estimate the combined probability
of all robbers' locations.

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
import matplotlib.animation as animation
from scipy import stats

from cops_and_robots.map_tools.layer import Layer
from cops_and_robots.fusion.gaussian_mixture import GaussianMixture
import itertools


class ProbabilityLayer(Layer):
    """A probabilistic distribution representing the target position.

    .. image:: img/classes_Probability_Layer.png

    Parameters
    ----------
    grid_size : float, optional
        The side length for each square cell in discretized probability map's
        cells. Defaults to 0.2.
    **kwargs
        Keyword arguments given to the ``Layer`` superclass.

    """
    def __init__(self, filter_, grid_size=0.2, z_levels=100, alpha=0.6,
                 colorbar_visible=False, show_ellipses=False,
                **kwargs):
        super(ProbabilityLayer, self).__init__(alpha=alpha, **kwargs)
        self.filter = filter_
        self.grid_size = grid_size  # in [m/cell]
        self.z_levels = z_levels
        self.colorbar_visible = colorbar_visible

        self.X, self.Y = np.mgrid[self.bounds[0]:self.bounds[2]:self.grid_size,
                                  self.bounds[1]:self.bounds[3]:self.grid_size]
        self.pos = np.empty(self.X.shape + (2,))
        self.pos[:, :, 0] = self.X
        self.pos[:, :, 1] = self.Y
        self.show_ellipses = show_ellipses

        # if colorbar_visible:
        # generate new axis for colorbar

    def plot(self, probability=None, **kwargs):
        """Plot the pseudo colormesh representation of probabilty.

        Parameters
        ----------
        probability : [multiple]
            Any probability with a 2D pdf.

        Returns
        -------
        QuadMesh
            The scatter pseudo colormesh data.
        """
        if probability is None:
            probability = self.filter.probability

        levels = np.linspace(0, np.max(probability.pdf(self.pos, dims=[0,1])),
                             self.z_levels)
        self.contourf = self.ax.contourf(self.X, self.Y, probability.pdf(self.pos, dims=[0,1]),
                           cmap=self.cmap, alpha=self.alpha, levels=levels, antialiased=True,
                           **kwargs)

        if self.show_ellipses:
            if hasattr(self.filter.probability, 'camera_viewcone'):
                poly = self.filter.probability.camera_viewcone
            else:
                poly = None
            self.ellipse_patches = probability.plot_ellipses(ax=self.ax,
                                                             poly=poly)
        # if colorbar_visible:
        #     self.cbar = plt.colorbar(p)

    def update(self, i=0):
        """Remove previous contour and replot new contour.
        """
        logging.debug('Probability Layer update {}'.format(i))
        # Test stub for the call from __main__
        if hasattr(self, 'test_probability'):
            self.filter.probability = next(self.test_probability)

        # Try to remove previous contourf and replot
        self.remove()
        self.plot()

        # if colorbar_visible:
        #    return self.contourf, self.cbar

        return self.contourf

    def remove(self):
        if hasattr(self, 'contourf'):
            for collection in self.contourf.collections:
                collection.remove()
            del self.contourf

        if hasattr(self, 'ellipse_patches'):
            for patch in self.ellipse_patches:
                patch.remove()
            del self.ellipse_patches

if __name__ == '__main__':
    d = GaussianMixture(1,[0, 0],[[1,0],[0,1]])
    filter_ = type('test', (object,), {'probability': d})()
    pl = ProbabilityLayer(d, z_levels=50, alpha=1)

    test_probability = []
    test_probability.append(GaussianMixture(1,[2, 0],[[1,0],[0,1]]))
    test_probability.append(GaussianMixture(1,[1, 1],[[1,0],[0,1]]))
    test_probability.append(GaussianMixture(1,[0, 2],[[1,0],[0,1]]))
    test_probability.append(GaussianMixture(1,[-1, 1],[[1,0],[0,1]]))
    test_probability.append(GaussianMixture(1,[-2, 0],[[1,0],[0,1]]))
    test_probability.append(GaussianMixture(1,[-1, -1],[[1,0],[0,1]]))
    test_probability.append(GaussianMixture(1,[0, -2],[[1,0],[0,1]]))
    test_probability.append(GaussianMixture(1,[1, -1],[[1,0],[0,1]]))
    test_probability.append(GaussianMixture(1,[2, 0],[[1,0],[0,1]]))
    pl.test_probability = itertools.cycle(test_probability)

    ani = animation.FuncAnimation(pl.fig, pl.update,
        frames=xrange(100),
        interval=50,
        repeat=True,
        blit=False)

    plt.show()
