#!/usr/bin/env python
"""Provides a discrete particle representation of robot locations.

When using the *discrete* ``fusion_engine`` type, the particle
layer is used to represent the distribution of a cop's expecation of
robber locations. One particle layer exists per robot, with n particles
per particle layer, to estimate the collected probability of all
robbers' locations.

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

import logging

from cops_and_robots.map_tools.layer import Layer


class ParticleLayer(Layer):
    """Visually represents a collection of particles.

    :param particle_size: marker size of each particle.
    :type particle_size: positive int.
    :param alpha: marker transparency of each particle.
    :type alpha: float between 0 and 1.
    :param line_weight: marker border weight for each particle.
    :type line_weight: positive float.
    :param colorbar_visible: whether or not to show the colorbar.
    :type colorbar_visible: bool.
    """
    def __init__(self, particle_size=200, alpha=0.3, line_weight=0,
                 colorbar_visible=False, n_particles=2000, **kwargs):
        super(ParticleLayer, self).__init__(alpha=alpha, **kwargs)
        self.particle_size = particle_size
        self.colorbar_visible = colorbar_visible
        self.line_weight = line_weight
        self.n_particles = n_particles

    def plot(self, robber_names, fusion_engine, **kwargs):
        """Plot the particles as a scatter plot.

        :param particle_filter: a collection of particles to be shown.
        :type particle_filter: ParticleFilter.
        :returns: the scatter plot data.
        :rtype: list ofPathCollection.
        """
        for name in robber_names:
            if fusion_engine.type == 'discrete':
                particle_filter = fusion_engine.filters[name]
            else:
                raise ValueError('Fusion engine type must be discrete.')
            p = self.ax.scatter(particle_filter.particles[:, 0],
                                particle_filter.particles[:, 1],
                                c=particle_filter.particles[:, 2],
                                cmap=self.cmap,
                                s=self.particle_size,
                                lw=self.line_weight,
                                alpha=self.alpha,
                                marker='.',
                                vmin=0,
                                vmax=1,
                                **kwargs
                                )
        return p
