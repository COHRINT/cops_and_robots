#!/usr/bin/env python
"""Provides an abstracted interface to multiple types of data fusion.

By dictating the type of the fusion_engine, a cop using the fusion
engine doesn't need to call specific functions for particle filters or
Gaussian Mixture Models -- it asks the fusion engine for what it wants
(generally a goal pose for the cop's planner) and lets the fusion
engine call whichever type of data fusion it wants to use.

Note:
    Only cop robots have fusion engines (for now). Robbers may get
    smarter in future versions, in which case this would be owned by
    the ``robot`` module instead of the ``cop`` module.

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``particle_filter`` as one method to represent information.
        2. ``gaussian_mixture_model`` as another method.
        3. ``feasible_layer`` to generate feasible particles and/or
           probabilities.
        4. ``shape_layer`` to ground the human sensor's output.
"""

from __future__ import division

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

from cops_and_robots.robo_tools.fusion.particle_filter import ParticleFilter
from cops_and_robots.robo_tools.fusion.gaussian_mixture_model import \
    GaussianMixtureModel


class FusionEngine(object):
    """

    The fusion engine tracks each robber, as well as a *combined*
    representation of the average target estimate.

    :param type_: 'discrete' or 'continuous'.
    :param type_: String.
    :param :
    :type :
    """
    def __init__(self,
                 type_,
                 missing_robber_names,
                 feasible_layer,
                 shape_layer,
                 motion_model='simple',
                 total_particles=4000):
        super(FusionEngine, self).__init__()

        self.type = type_
        self.filters = {}
        self.GMMs = {}
        self.missing_robber_names = missing_robber_names
        self.shape_layer = shape_layer

        n = len(missing_robber_names)
        if n > 1:
            particles_per_filter = int(total_particles / (n + 1))
        else:
            particles_per_filter = total_particles

        if self.type == 'discrete':
            for i, name in enumerate(missing_robber_names):
                self.filters[name] = ParticleFilter(name,
                                                    feasible_layer,
                                                    motion_model,
                                                    particles_per_filter)
            self.filters['combined'] = ParticleFilter('combined',
                                                      feasible_layer,
                                                      motion_model,
                                                      particles_per_filter)
        elif self.type == 'continuous':
            for i, name in enumerate(missing_robber_names):
                self.GMMs[name] = GaussianMixtureModel(name, feasible_layer)
            self.GMMs['combined'] = GaussianMixtureModel('combined',
                                                         feasible_layer)
        else:
            raise ValueError("FusionEngine must be of type 'discrete' or "
                             "'continuous'.")

    def update(self, current_pose, sensors, robbers):
        """Update fusion_engine agnostic to fusion type.

        :param fusion_engine: a probabalistic target tracker.
        :type fusion_engine: FusionEngine.
        :param feasible_layer: a map of feasible regions.
        :type feasible_layer: FeasibleLayer.
        :returns: goal pose as x,y,theta.
        :rtype: list of floats.
        """

        # Update camera values (viewcone, selected zone, etc.)
        for sensorname,sensor in sensors.iteritems():
            if sensorname == 'camera':
                sensor.update(current_pose, self.shape_layer)

        # Update probabilities (particle and/or GMM)
        for robber in robbers.values():
            if self.type == 'discrete':
                if not self.filters[robber.name].finished:
                    self.filters[robber.name].update(sensors['camera'],
                                                     robber.pose)
                    if robber.status == 'detected':
                        logging.info('{} detected!'.format(robber.name))
                        self.filters[robber.name].robber_detected(robber.pose)
            else:
                self.GMMs[robber.name].update()
                if robber.status == 'detected':
                    self.GMMs[robber.name].robber_detected(robber.pose)

            # Chop down list of missing robber names if one was captured
            if robber.status == 'captured':
                    logging.info('{} captured!'.format(robber.name))
                    self.missing_robber_names.remove(robber.name)

        self.update_combined()

    def update_combined(self):
        if self.type == 'discrete':

            # Remove all particles from combined filter
            self.filters['combined'].particles = \
                np.random.uniform(0, 0, (1, 3))

            # Add all particles from missing robots to combined filter
            for i, name in enumerate(self.missing_robber_names):
                self.filters['combined'].n_particles += \
                    self.filters[name].n_particles
                self.filters['combined'].particles = \
                    np.append(self.filters['combined'].particles,
                              self.filters[name].particles,
                              axis=0)
            self.filters['combined'].n_particles = \
                len(self.filters['combined'].particles)
