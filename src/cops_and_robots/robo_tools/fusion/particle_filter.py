#!/usr/bin/env python
"""Provides a particle cloud to estimate target robber locatons.

The particle filter takes a brute-force approach to estimating target
parameters (starting with 2D position, but planned to move to 3D
pose, velocity, and anything else estimatable). It does this by
generating n particles, each one as a separate discrete potential
model of the target. Each particle has estimated parameters (i.e, 2D
position), weights (how important that particle is), and probabilities
(how likely the estimated value is). For many cases, weights and
probabilities are interchangable.

"""
__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

from pylab import *

import logging
import math
import numpy as np

from shapely.geometry import Point


class ParticleFilter(object):
    """Particle-based representation of target locations.

    .. image:: img/classes_Particle_Filter.png

    Parameters
    ----------
    target : str
        Name of the target tracked by the particle filter ('combined' for
        all targets).
    feasible_layer : FeasibleLayer
        A layer object providing both permissible point regions for any object
        and permissible pose regions for any robot with physical dimensions.
    motion_model : {'stationary','clockwise','counterclockwise','random walk'},
        optional
        The motion model used to update each particle.
    n_particles : int, optional
        The number of particles this particle filter has. Default is 500.

    """
    def __init__(self, target, feasible_layer, motion_model='stationary',
                 n_particles=500):
        self.target = target
        self.bounds = feasible_layer.bounds
        self.n_particles = n_particles
        self.motion_model = motion_model
        self.feasible_layer = feasible_layer
        self.finished = False

        self.generate_particles()

    def generate_particles(self):
        """Generate uniformly distributed particles.
        """
        bounds = self.feasible_layer.bounds
        px = np.random.uniform(bounds[0], bounds[2], self.n_particles)
        py = np.random.uniform(bounds[1], bounds[3], self.n_particles)
        pts = np.column_stack((px, py))

        for i, pt in enumerate(pts):
            feasible_particle_generated = False
            while not feasible_particle_generated:
                point = Point(pt[0], pt[1])
                if self.feasible_layer.point_region.contains(point):
                    feasible_particle_generated = True
                else:
                    pt[0] = np.random.uniform(bounds[0], bounds[2])
                    pt[1] = np.random.uniform(bounds[1], bounds[3])
        self.particles = np.column_stack((pts, np.ones(self.n_particles)
                                          / self.n_particles))

    def update(self, camera, target_pose, human=None):
        """Move particles (if mobile) and update probabilities.

        Parameters
        ----------
        camera : Camera
            A camera sensor object to update the particle weights and
            probabilities.
        target_pose : array_like
            The target's current [x, y, theta] in [m,m,degrees].
        human : Human, optional
            A human sensor object to update the particle weights and
            probabilities. Default is `None` for no update.

        """
        if self.finished:
            return

        self.update_particle_motion()
        self._camera_update(camera)
        self._human_update(human)

        # <>TODO: Resample particles
        # self.resample()

    def update_particle_motion(self, step_dist=0.05):
        """Update one step in the particles' motion.

        Parameters
        ----------
        step_dist : float, optional
            A scaling factor for how large the update to the particles'
            positions is. Default is 0.05.
        """
        if self.motion_model == 'random walk':
            self.particles[:, 0:2] += np.random.uniform(-2 * step_dist,
                                                        2 * step_dist,
                                                        (self.n_particles, 2),
                                                        )
        elif self.motion_model == 'counterclockwise':
            for particle in self.particles:
                x, y = particle[0:2]
                mag = math.sqrt(x ** 2 + y ** 2) + \
                    np.random.uniform(-step_dist, step_dist)
                angle = math.atan2(y, x) + \
                    np.random.uniform(step_dist / 5, step_dist)
                particle[0] = mag * math.cos(angle)
                particle[1] = mag * math.sin(angle)
        elif self.motion_model == 'clockwise':
            for particle in self.particles:
                x, y = particle[0:2]
                mag = math.sqrt(x ** 2 + y ** 2) + \
                    np.random.uniform(-step_dist, step_dist)
                angle = math.atan2(y, x) + \
                    np.random.uniform(-step_dist, step_dist / 5)
                particle[0] = mag * math.cos(angle)
                particle[1] = mag * math.sin(angle)

    def _camera_update(self, camera):
        """Update the particle filter's values from a camera update event.

        Parameters
        ----------
        camera : Camera
            A camera sensor object.

        """
        camera.detect('particle', self.particles)

    def _human_update(self, human):
        """Update the particle filter's values from a human sensor update.

        Parameters
        ----------
        human : Human
            A human sensor object.

        """
        if human.target in [self.target, 'nothing', 'a robber']:
            motion_model = human.detect(self.particles)
            if human.target in [self.target, 'a robber']:
                self.motion_model = motion_model
            elif motion_model == self.motion_model:
                self.motion_model = 'stationary'

    def resample(self):
        """Sample or resample distribution to generate new particles

        """
        # sample distribution
        # <>Constrain particle limits to map limits
        # <>Always resample self.n_particles
        self.particles = self.prob.rvs(size=self.n_particles)

        # recalculate particles
        self.particle_probs = [1 / self.n_particles
                               for i in range(0, self.n_particles)]

    def robber_detected(self, robber_pose):
        """Update the particle filter for a detected robber.
        """

        # <>TODO: Figure out better strategy when robber detected

        # Find closest particle to target
        # robber_pt = robber_pose[0:2]
        # dist = [math.sqrt((pt[0] - robber_pt[0]) ** 2
        #                   + (pt[1] - robber_pt[1]) ** 2)
        #         for pt in self.particles[:, 0:2]]
        # index = dist.index(min(dist))

        # Set all other particles to 0 probability
        # self.particles[:, 2] = 0
        # self.particles[index] = 1

        self.particles = np.zeros([1, 3])
        self.finished = True


if __name__ == '__main__':
    # Run a test probability layer creation
    pass
