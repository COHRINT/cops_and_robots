#!/usr/bin/env python
"""Provides a common base class for all map layers.

Since many layers share parameters and functions, the ``layer`` module 
defines these in one place, allowing all layers to use it as a 
superclass.

Required Knowledge:
    This module and its classes do not need to know about any other 
    parts of the cops_and_robots parent module.
"""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import matplotlib.pyplot as plt

class Layer(object):
    """A collection of generic layer parameters and functions.

    :param bounds: Map boundaries as [xmin,ymin,xmax,ymax] in [m].
    :type bounds: List of floats.
    :param visible: Whether or not the layer should be plotted.
    :type visible: bool.
    :param target: Name of the target tracked (for some layers only).
    :type target: String.
    :param ax: Axes to be used for plotting this layer.
    :type ax: Axes.
    :param alpha: Transparency of all elements of this layer.
    :type alpha: float.
    :param cmap_str: Name of the color map to be used.
    :type cmap_str: String.
    """
    def __init__(self,bounds=[-5,-5,5,5],visible=True,target=None,
                 ax=None,alpha=0.8,cmap_str='jet'):
        if ax == None:
            ax = plt.gca()

        self.bounds = bounds #[xmin,ymin,xmax,ymax] in [m]
        self.visible = visible
        self.target = target #Name of target tracked by this layer
        self.alpha = alpha
        self.cmap = plt.cm.get_cmap(cmap_str)
