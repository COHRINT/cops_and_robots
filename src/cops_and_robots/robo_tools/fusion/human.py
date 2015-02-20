#!/usr/bin/env python
"""Provides a structured input interface for a human to the cop robot.

The human can provide some information that a robot can't easily
determine on its own (i.e. visual recognition of a target). Given a
proper GUI, the human can send information that the fusion engine will
merge with the robot's own sensors to generate a rich probability
distribution.

Currently the human uses a simple grounded, pre-defined codebook to
provide information. For example, the human can say, "I think Roy is
behind wall 3." from a pre-defined codebook that has all robbers (i.e.
Roy), spatial relationships (i.e. 'behind'), and physical groundings
(i.e. wall 3). This is the basic interface from which we will build
upon.

Note
----
    Only cop robots have human helpers (for now). Robbers may get
    human teammate upgreades in future versions, in which case this
    would be owned by the ``robot`` module instead of the ``cop`` module.

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

from shapely.geometry import Point

from cops_and_robots.robo_tools.fusion.sensor import Sensor


class Human(Sensor):
    """The human sensor, able to provide updates to a fusion engine.

    .. image:: img/classes_Human.png

    Parameters
    ----------
    shape_layer : ShapeLayer
        A layer object providing all the shapes in the map so that the human
        sensor can ground its statements.
    robber_names : list of str
        The list of all robbers, for the human to specify targets.
    detection_chance : float
        The human sensor's ability to correctly name a target, given that
        the target is within the human sensor's view -- that is, P(D=i|x).

        Note that this is the detection chance related to the "I think"
        statement, which can be increased if the human says "I know" instead.

    """
    def __init__(self, shape_layer, robber_names, detection_chance=0.6):
        self.update_rate = None
        self.has_physical_dimensions = False
        self.detection_chance = detection_chance
        self.certain_detection_chance = detection_chance + \
            (1 - detection_chance) * 0.8
        self.no_detection_chance = 0.01
        super(Human, self).__init__(self.update_rate,
                                    self.has_physical_dimensions,
                                    self.detection_chance)

        self.grounding_objects = shape_layer.shapes
        self.groundings = shape_layer.shapes.keys()
        self.groundings.sort()
        robber_names.sort()
        self.robber_names = ['nothing', 'a robber'] + robber_names
        self.certainties = ['think', 'know']
        self.relationships = ['behind', 'in front of', 'left of', 'right of']
        self.movements = ['stopped', 'moving CCW', 'moving CW',
                          'moving randomly']

        self.defaults = {'certainty': 'think',
                         'target': 'a robber',
                         'timing': 'is',
                         'relation': 'behind',
                         'grounding': 'wall 1'
                         }
        self.input_string = ''
        self.target = ''

    def detect(self, particles=None):
        """Update a fusion engine's probability from human sensor updates.

        Parameters
        ----------
        particles : array_like
            The particle list, assuming [x,y,p], where x and y are position
            data and p is the particle's associated probability. `None` if not
            using particles.
        """

        # Parse the input string
        for str_ in self.certainties:
            if str_ in self.input_string:
                certainty = str_

        for str_ in self.relationships:
            if str_ in self.input_string:
                relation = str_

        for str_ in self.groundings:
            if str_ in self.input_string:
                grounding = str_

        for str_ in self.movements:
            if str_ in self.input_string:
                movement = str_

        # Translate relation to zone label
        if relation == 'behind':
            zone_label = 'back'
        elif relation == 'in front of':
            zone_label = 'front'
        elif relation == 'left of':
            zone_label = 'left'
        elif relation == 'right of':
            zone_label = 'right'
        zone = self.grounding_objects[grounding].zones_by_label[zone_label]

        # Translate movement to motion model
        if movement == 'stopped':
            motion_model = 'stationary'
        elif movement == 'moving CW':
            motion_model = 'clockwise'
        elif movement == 'moving CCW':
            motion_model = 'counterclockwise'
        elif movement == 'moving randomly':
            motion_model = 'random walk'

        # <>TODO consider probabilities as well
        self.detect_particles(particles, zone, certainty)

        return motion_model

    def detect_particles(self, particles, zone, certainty):
        """Update particles based on sensor model.

        Parameters
        ----------
        particles : array_like
            The particle list, assuming [x,y,p], where x and y are position
            data and p is the particle's associated probability. `None` if not
            using particles.
        zone : str
            The zone specified by the human sensor.
        certainty : float
            The certainty specified by the human sensor.

        """
        # Update particle probabilities in view cone
        for i, particle in enumerate(particles):

            # Negative information
            if self.target == 'nothing':
                if zone.contains(Point(particle[0:2])):
                    if certainty == 'know':
                        particles[i, 2] *= (1 - self.certain_detection_chance)
                        print(particles[i, 2])
                    else:
                        particles[i, 2] *= (1 - self.detection_chance)
                        print(particles[i, 2])
            else:
                # Positive information
                if zone.contains(Point(particle[0:2])):
                    if certainty == 'know':
                        particles[i, 2] *= self.certain_detection_chance
                    else:
                        particles[i, 2] *= self.detection_chance
                else:
                    particles[i, 2] *= self.no_detection_chance

        # Renormalize
        particles[:, 2] /= sum(particles[:, 2])
