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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import itertools

from cops_and_robots.map_tools.layer import Layer


class ParticleLayer(Layer):
    """Visually represents a collection of particles.

    .. image:: img/classes_Particle_Layer.png

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
    def __init__(self, filter_, particle_size=200,
                 colorbar_visible=False, alpha=0.3,
                 **kwargs):
        super(ParticleLayer, self).__init__(alpha=alpha, **kwargs)
        self.filter = filter_
        self.particle_size = particle_size
        self.colorbar_visible = colorbar_visible
        self.line_weight = 0
        self.color_gain = 400

    def plot(self, particles=None, **kwargs):
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
        if particles is None:
            particles = self.filter.particles

        if particles == 'Finished':
            print ' also done'

        self.scatter = self.ax.scatter(particles[:, 1],
                                       particles[:, 2],
                                       c=particles[:, 0] * self.color_gain,
                                       cmap=self.cmap,
                                       s=self.particle_size,
                                       lw=self.line_weight,
                                       alpha=self.alpha,
                                       marker='.',
                                       vmin=0,
                                       vmax=1,
                                       **kwargs
                                       )
        # <>TODO: Possible issue with scatter setting axis limits

    def update(self, i=0):
        """Remove previous scatter and replot new scatter
        """
        # Test stub for the call from __main__
        if hasattr(self, 'test_particles'):
            self.filter.particles = next(self.test_particles)

        # Remove previous scatter plot and replot
        self.remove()
        self.plot()

    def remove(self):
        if hasattr(self, 'scatter'):
            self.scatter.remove()
            del self.scatter

def main():
    x = np.random.uniform(0, 10, 100)
    y = np.random.uniform(0, 10, 100)
    pts = np.column_stack((x, y))
    probs = np.ones(100) / 100
    particles = np.column_stack((probs, pts))
    filter_ = type('test', (object,), {'particles': particles})()

    pal = ParticleLayer(filter_, bounds=[0, 0, 10, 10])
    pal.color_gain = 1

    test_particles = {}
    for i in range(10):
        x = np.random.uniform(0, 10, 100)
        y = np.random.uniform(0, 10, 100)
        pts = np.column_stack((x, y))
        probs = np.random.uniform(0, 1, 100)
        # probs = np.ones(100) * .5
        particles = np.column_stack((probs, pts))
        test_particles['{}'.format(i)] = particles
    for i in range(10):
        i = i + 10
        test_particles['{}'.format(i)] = np.column_stack(([],[],[]))

    pal.test_particles = itertools.cycle(test_particles.values())

    ani = animation.FuncAnimation(pal.fig, pal.update,
        frames=xrange(100),
        interval=100,
        repeat=True,
        blit=False)

    plt.show()


if __name__ == '__main__':
    main()
