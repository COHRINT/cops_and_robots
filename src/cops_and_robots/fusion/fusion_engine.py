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

from cops_and_robots.fusion.filter import GridFilter
from cops_and_robots.fusion.particle_filter import ParticleFilter
from cops_and_robots.fusion.gauss_sum_filter import \
    GaussSumFilter
from cops_and_robots.fusion.gaussian_mixture import GaussianMixture


class FusionEngine(object):
    """A collection of filters to estimate and fuse data about the targets.

    The fusion engine tracks each robber, as well as a `combined`
    representation of the average target estimate.

    .. image:: img/classes_Fusion_Engine.png

    Parameters
    ----------
    probability_type : {'grid','particle','gauss sum'}
        The type of filter to be used.
    missing_robber_names : list of str
        The list of all robbers, to create one filter per robber.
    feasible_layer : FeasibleLayer
        A layer object providing both permissible point regions for any object
        and permissible pose regions for any robot with physical dimensions.
    motion_model : {'stationary','clockwise','counterclockwise','random walk'},
        optional
        The motion model used to update the filter.
    total_particles : int, optional
        The number of particles used for a particle filter. This is the number
        used for the `combined` model, which is distributed among all other
        models. Default is 2000.

    """

    def __init__(self,
                 probability_type,
                 missing_robber_names,
                 feasible_layer,
                 motion_model='stationary',
                 total_particles=2000,
                 rosbag_process=None,
                 use_velocity=True,
                 use_STM=True,
                 ):

        self.probability_type = probability_type
        self.filters = {}
        self.missing_robber_names = missing_robber_names

        n = len(missing_robber_names)

        if self.probability_type == 'grid':
            for i, name in enumerate(missing_robber_names):
                self.filters[name] = GridFilter(name, feasible_layer, 
                                                rosbag_process=rosbag_process,
                                                use_STM=use_STM,
                                                )
            if len(missing_robber_names) > 1:
                self.filters['combined'] = GridFilter('combined', 
                                                      feasible_layer,
                                                      use_STM=use_STM,
                                                      )

        elif self.probability_type == 'particle':
            if n > 1:
                particles_per_filter = int(total_particles / (n + 1))
            else:
                particles_per_filter = total_particles

            for i, name in enumerate(missing_robber_names):
                self.filters[name] = ParticleFilter(name,
                                                    feasible_layer,
                                                    motion_model,
                                                    particles_per_filter)
            self.filters['combined'] = ParticleFilter('combined',
                                                      feasible_layer,
                                                      motion_model,
                                                      particles_per_filter)
        elif self.probability_type == 'gauss sum':
            for name in missing_robber_names:
                self.filters[name] = GaussSumFilter(target_name=name,
                                                    feasible_layer=feasible_layer, 
                                                    rosbag_process=rosbag_process)
            if len(missing_robber_names) > 1:
                self.filters['combined'] = GaussSumFilter('combined', feasible_layer)
        else:
            raise ValueError("FusionEngine must be of type 'grid', 'particle'"
                             "or 'gauss sum'.")

    def update(self, robot_pose, sensors, robbers, save_file=None):
        """Update fusion_engine agnostic to fusion type.

        Parameters
        ----------
        robot_pose : array_like
            The robot's current [x, y, theta] in [m,m,degrees].
        sensors : dict
            A collection of all sensors to be updated.
        robbers :
            A collection of all robber objects.
        """
        # Update camera values (viewcone, selected zone, etc.)
        for sensorname, sensor in sensors.iteritems():
            #<>TODO: Fix this - it shouldn't be in the fusion engine's update
            if sensorname == 'camera':
                sensor.update_viewcone(robot_pose)

        # Update the filter probabilities
        for robber in robbers.values():
            if not self.filters[robber.name].finished:

                # Update position filters
                self.filters[robber.name].update(sensors['camera'],
                                                 sensors['human'],
                                                 )

        if len(self.missing_robber_names) > 1:
            self._update_combined(sensors, robbers)
        sensors['human'].clear_measurement()

    def _update_combined(self, sensors, robbers):
        # Fuse grid-based filters
        for label, filter_ in self.filters.iteritems():
 
            # Fuse it in with the rest
            try:
                prob *= filter_.probability.prob
            except NameError:
                prob = filter_.probability.copy().prob
        prob /= prob.sum()
        self.filters['combined'].probability.prob = prob

    def _update_combined_OLD(self, sensors, robbers):
        """Update the `combined` filter.

        Parameters
        ----------
        sensors : dict
            A collection of all sensors to be updated.
        """
        if self.probability_type == 'particle':

            # Remove all particles from combined filter
            # <>TODO: correct number of particles based on state
            self.filters['combined'].particles = np.zeros((1, 5))

            # Add all particles from missing robots to combined filter
            for robber in robbers.values():
                self.filters['combined'].n_particles += \
                    self.filters[robber.name].n_particles
                self.filters['combined'].particles = \
                    np.append(self.filters['combined'].particles,
                              self.filters[robber.name].particles,
                              axis=0)
            self.filters['combined'].n_particles = \
                len(self.filters['combined'].particles)

            # Reset the human sensor
            sensors['human'].utterance = ''
            sensors['human'].target = ''
        else:

            # Pre-allocate parameter arrays
            num_mixands = 0
            all_done = True
            for label, filter_ in self.filters.iteritems():
                if label == 'combined' or filter_.finished:
                    continue
                all_done = False
                num_mixands += filter_.probability.num_mixands
                # <>TODO: Don't find ndims each time
                ndims = filter_.probability.ndims
            if all_done:
                # <>TODO: Fix hardcoded dimensions, clean it up
                gm = GaussianMixture(1, [0, 0], 0.01 * np.eye(2))
                self.filters['combined'].probability = gm
                return
            weights = np.empty((num_mixands))
            means = np.empty((num_mixands, ndims))
            covariances = np.empty((num_mixands, ndims, ndims))

            # Load parameter arrays
            i = 0
            for label, filter_ in self.filters.iteritems():
                if label == 'combined' or filter_.finished:
                    continue
                size = filter_.probability.num_mixands
                weights[i:i + size] = filter_.probability.weights
                means[i:i + size] = filter_.probability.means
                covariances[i:i + size] = filter_.probability.covariances
                i += size

            combined_gm = GaussianMixture(weights, means, covariances,
                                          max_num_mixands=num_mixands)
            self.filters['combined'].probability = combined_gm
