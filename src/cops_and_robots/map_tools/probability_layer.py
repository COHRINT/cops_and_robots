#!/usr/bin/env python
"""Provides a continuous probability representation of robot locations.

When using the *continuous* ``fusion_engine`` type, the probability 
layer is used to represent the distribution of a cop's expecation of
robber locations. One probability layer exists per robot, as well as 
one additional probability layer to estimate the combined probability 
of all robbers' locations.

Required Knowledge:
    This module and its classes needs to know about the following 
    other modules in the cops_and_robots parent module:
        1. ``layer`` for generic layer parameters and functions.
        2. ``fusion_engine`` to provide the probability distribution.
"""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pylab import *
from shapely.geometry import Point

from cops_and_robots.map_tools.layer import Layer


class ProbabilityLayer(Layer):
    """A probabilistic distribution representing the target position.

    :param cell_size: size of the discretized probability map's cells.
    :type cell_size: positive float.
    :param colorbar_visible: whether or not to show the colorbar.
    :type colorbar_visible: bool.
    """
    def __init__(self,cell_size=0.2,**kwargs):
        super(ProbabilityLayer, self).__init__(**kwargs)
        self.cell_size = cell_size #in [m/cell]
        self.colorbar_visible = colorbar_visible


        xlin = np.linspace(bounds[0],bounds[2],100)
        ylin = np.linspace(bounds[1],bounds[3],100)
        self.X, self.Y = np.meshgrid(xlin,ylin)
        self.pos = np.empty(self.X.shape + (2,))
        self.pos[:, :, 0] = self.X
        self.pos[:, :, 1] = self.Y

        self.MAP = [0, 0] #[m] point of maximum a posteriori

        self.prob = stats.multivariate_normal([0,0], [[10,0], [0,10]])
        
    def plot(self,GMM,**kwargs):
        """Plot the pseudo colormesh representation of probabilty.

        :param GMM: gaussian mixture model as a continuous probabilty.
        :type GMM: GaussianMixtureModel.
        :returns: the scatter pseudo colormesh data.
        :rtype: QuadMesh.
        """
        p = plt.pcolormesh(self.X,self.Y,GMM.prob.pdf(self.pos),
                       cmap=self.cmap,alpha=self.alpha,**kwargs)
        if colorbar_visible:
            cb = plt.colorbar(p)    
        return p
    

if __name__ == '__main__':
    #Run a test probability layer creation        
    pass