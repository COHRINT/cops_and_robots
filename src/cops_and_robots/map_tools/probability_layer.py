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
from scipy import stats
from pylab import *

from cops_and_robots.map_tools.layer import Layer


class ProbabilityLayer(Layer):
    """A probabilistic distribution representing the target position.

    .. image:: img/classes_Probability_Layer.png

    Parameters
    ----------
    cell_size : float, optional
        The side length for each square cell in discretized probability map's
        cells. Defaults to 0.2.
    **kwargs
        Keyword arguments given to the ``Layer`` superclass.

    """
    def __init__(self, cell_size=0.2, **kwargs):
        super(ProbabilityLayer, self).__init__(**kwargs)
        self.cell_size = cell_size  # in [m/cell]
        self.colorbar_visible = colorbar_visible

        xlin = np.linspace(bounds[0], bounds[2], 100)
        ylin = np.linspace(bounds[1], bounds[3], 100)
        self.X, self.Y = np.meshgrid(xlin, ylin)
        self.pos = np.empty(self.X.shape + (2,))
        self.pos[:, :, 0] = self.X
        self.pos[:, :, 1] = self.Y

        self.MAP = [0, 0]  # [m] point of maximum a posteriori

        self.prob = stats.multivariate_normal([0, 0], [[10, 0], [0, 10]])

    def plot(self, gauss_sum, **kwargs):
        """Plot the pseudo colormesh representation of probabilty.

        Parameters
        ----------
        gauss_sum : GaussSum
            A Gaussian sum distribution.

        Returns
        -------
        QuadMesh
            The scatter pseudo colormesh data.
        """
        p = plt.pcolormesh(self.X, self.Y, gauss_sum.prob.pdf(self.pos),
                           cmap=self.cmap, alpha=self.alpha, **kwargs)
        if colorbar_visible:
            cb = plt.colorbar(p)
        return p, cb
