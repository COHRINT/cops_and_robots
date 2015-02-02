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

Note:
    Only cop robots have human helpers (for now). Robbers may get
    human teammate upgreades in future versions, in which case this
    would be owned by the ``robot`` module instead of the ``cop`` module.

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``sensor`` for generic sensor parameters and functions,
           such as update rate and detection chance.
        2. ``shape_layer`` to ground the human's provided information.
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
import re

from cops_and_robots.robo_tools.fusion.sensor import Sensor


class Human(Sensor):
    """docstring for Human
    :param shape_layer: the groundings for the human's updates.
    :type shape_layer: ShapeLayer.
    """
    def __init__(self, map_objs, robber_names, detection_chance=0.6):
        self.update_rate = None
        self.has_physical_dimensions = False
        self.detection_chance = detection_chance
        super(Human, self).__init__(self.update_rate, self.has_physical_dimensions,
                                    detection_chance)

        self.groundings = map_objs
        self.targets = ['nothing', 'a robber'] + robber_names
        self.defaults = {'certainty': 'think',
                         'target': 'a robber',
                         'timing': 'is',
                         'relative': 'behind',
                         'grounding': 'wall 1'
                         }

    def detect(self,particles):
        human_string = self.human_string
        human_string = 'I think Roy is behind wall 1.'
        logging.info('Human says: {}'.format(human_string))

        human_string = human_string.split(' ',1)[1]
        certainty = human_string.split(' ')[0]
        human_string = human_string.split(' ',1)[1]

        i = human_string.index('is')
        target = human_string[:i-1]
        human_string = human_string[i:]
        human_string = human_string.split(' ',1)[1]

        grounding = human_string.split(' ',1)[1]
        grounding = grounding[:-1]
        human_string = human_string.split(' ',1)[0]

        relative = human_string

        self.human_string = ''

        zone = self.groundings[grounding].zones_by_label[relative]
        detect_particles(particles,zone)

    def detect_particles(self,particles,zone):
        # Update particle probabilities in view cone
        for i, particle in enumerate(particles):
            if zone.contains(Point(particle[0:2])):
                particles[i, 2] *= (1 - self.detection_chance)

        # Renormalize
        particles[:, 2] /= sum(particles[:, 2])
