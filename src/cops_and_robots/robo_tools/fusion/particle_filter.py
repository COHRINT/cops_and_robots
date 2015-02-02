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

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``feasible_layer`` to generate feasible particles.
        2. ``camera`` to update particles from sensor measurements.
        3. ``human`` to update particles from sensor measurements.
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

    :param target:
    :type target:
    :param feasible_layer: a map of feasible regions.
    :type feasible_layer: FeasibleLayer.
    :param n_particles:
    :type n_particles: integer.
    :param motion_model:
    :type motion_model:
    """
    def __init__(self, target, feasible_layer, motion_model='stationary',
                 n_particles=500):
        self.target = target  # MapObj of target for specific map layer
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

    def update(self, camera, target_pose):
        """Move particles (if mobile) and update probabilities.
        """
        if self.finished:
            return

        self.update_particle_motion()

        self.camera_update(camera, target_pose)
        # self.human_update()

        # self.resample()

    def camera_update(self, camera, target_pose):
        camera.detect('discrete', self.particles)

    def human_update(self):
        # <>TODO: break human out into seperate class
        pass

    def update_particle_motion(self, step_dist=0.05):

        if self.motion_model == 'random walk':
            self.particles[:, 0:2] += np.random.uniform(-2 * step_dist,
                                                        2 * step_dist,
                                                        (self.n_particles, 2),
                                                        )
        elif self.motion_model == 'clockwise':
            for particle in self.particles:
                x, y = particle[0:2]
                mag = math.sqrt(x ** 2 + y ** 2) + \
                    np.random.uniform(-step_dist, step_dist)
                angle = math.atan2(y, x) + \
                    np.random.uniform(step_dist / 5, step_dist)
                particle[0] = mag * math.cos(angle)
                particle[1] = mag * math.sin(angle)
        elif self.motion_model == 'counterclockwise':
            for particle in self.particles:
                x, y = particle[0:2]
                mag = math.sqrt(x ** 2 + y ** 2) + \
                    np.random.uniform(-step_dist, step_dist)
                angle = math.atan2(y, x) + \
                    np.random.uniform(-step_dist, step_dist / 5)
                particle[0] = mag * math.cos(angle)
                particle[1] = mag * math.sin(angle)

    def resample(self):
        '''Sample or resample distribution to generate new particles'''

        # sample distribution
        # <>Constrain particle limits to map limits
        # <>Always resample self.n_particles
        self.particles = self.prob.rvs(size=self.n_particles)

        # recalculate particles
        self.particle_probs = [1 / self.n_particles
                               for i in range(0, self.n_particles)]

    def robber_detected(self, robber_pose):

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
