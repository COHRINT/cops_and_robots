#!/usr/bin/env python
"""Provides a common base class for all map layers.

Since many layers share parameters and functions, the ``layer`` module
defines these in one place, allowing all layers to use it as a
superclass.

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

import matplotlib.pyplot as plt


class Layer(object):
    """A collection of generic layer parameters and functions.

    .. image:: img/classes_Layer.png

    Parameters
    ----------
    bounds : array_like, optional
        Map boundaries as [xmin,ymin,xmax,ymax] in [m]. Defaults to
        [-5, -5, 5, 5].
    visible : bool, optional
        Whether or not the layer is shown when plotting.
    target : str, optional
        Name of target tracked by this layer. Defaults to `''`.
    ax : axes handle, optional
        The axes to be used for plotting. Defaults to current axes.
    alpha : float, optional
        The layer's transparency, from 0 to 1. Defaults to 0.8.
    cmap_str : str, optional
        The colormap string for the layer. Defaults to `'jet'`.

    """
    def __init__(self, bounds=[-5, -5, 5, 5], visible=True, target='',
                 ax=None, alpha=0.8, cmap_str='jet'):
        if ax is None:
            ax = plt.gca()
        self.ax = ax

        self.bounds = bounds  # [xmin,ymin,xmax,ymax] in [m]
        self.visible = visible
        self.target = target
        self.alpha = alpha
        self.cmap = plt.cm.get_cmap(cmap_str)
