#!/usr/bin/env python
"""Provides a discrete particle representation of robot locations.

When using the *discrete* ``fusion_engine`` type, the particle
layer is used to represent the distribution of a cop's expecation of
robber locations. One particle layer exists per robot, with n particles
per particle layer, to estimate the collected probability of all
robbers' locations.

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

    Parameters
    ----------
    particle_size : int, optional
        The marker size of each particle. Default is 200.
    colorbar_visible : bool, optional
        Whether to show the colorbar object. Default it `False`.
    n_particles : int, optional
        Number of particles to display. Default is 2000.
    **kwargs
        Keyword arguments given to the ``Layer`` superclass.

    """
    def __init__(self, particle_size=200, colorbar_visible=False,
                 n_particles=2000, **kwargs):
        super(ParticleLayer, self).__init__(**kwargs)
        self.particle_size = particle_size
        self.colorbar_visible = colorbar_visible
        self.n_particles = n_particles  # <>TODO: get rid of this!

        self.line_weight = 0
        self.alpha = 0.5
        self.color_gain = 400

    def plot(self, robber_names, fusion_engine, **kwargs):
        """Plot the particles as a scatter plot.

        Parameters
        ----------
        robber_names : list of str
            The list of all robbers.
        fusion_engine : FusionEngine
            A robot's fusion engine.
        **kwargs
            Arguments passed to the scatter plot function.

        Returns
        -------
        list of PathCollection
            The scatter plot data.

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
