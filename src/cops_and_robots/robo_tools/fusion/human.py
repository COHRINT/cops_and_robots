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
Roy), spatial relations (i.e. 'behind'), and physical groundings
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
import numpy as np

from shapely.geometry import Point

from cops_and_robots.robo_tools.fusion.sensor import Sensor
from cops_and_robots.robo_tools.fusion.softmax import SoftMax, speed_model


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
    def __init__(self, map_=None, detection_chance=0.6):
        self.update_rate = None
        self.has_physical_dimensions = False
        # self.detection_chance = detection_chance
        # self.certain_detection_chance = detection_chance + \
        #     (1 - detection_chance) * 0.8
        sm = speed_model()
        self.motion_labels = sm.class_labels;
        self.detection_model = sm

        self.no_detection_chance = 0.01
        super(Human, self).__init__(self.update_rate,
                                    self.has_physical_dimensions)

        self.certainties = ['think', 'know']
        self.positivities = ['is', 'is not']
        self.relations = {'object': ['behind',
                                     'in front of',
                                     'left of',
                                     'right of',
                                     ],
                          'area': ['inside',
                                   'near',
                                   'outside'
                                   ]}
        self.movement_types = ['moving','stopped',]
        self.movement_qualities = ['slowly', 'moderately', 'quickly']

        self.groundings = {}
        self.groundings['area'] = map_.areas
        self.groundings['object'] = map_.objects + map_.cops
        self.target_names = ['nothing', 'a robot'] + map_.robbers.keys()
        self.utterance = ''

    def detect(self, target_name, type_="particle", particles=None, GMM=None):
        """Update a fusion engine's probability from human sensor updates.

        Parameters
        ----------
        particles : array_like
            The particle list, assuming [x,y,p], where x and y are position
            data and p is the particle's associated probability. `None` if not
            using particles.
        """

        if parse_utterance():
            logging.error('No utterance to parse!')

        # End detect loop if not the right target
        if self.target_name not in ['nothing', 'a robot', target_name]:
            return

        if any(r in self.utterance for r in self.relations['object']):
            translate_relation()
            self.detection_type = 'position (object)'
        if any(r in self.utterance for r in self.relations['area']):
            translate_relation()
            self.detection_type = 'position (area)'
        elif any(m in self.utterance for m in self.movements):
            translate_movement()
            self.detection_type = 'movement'
        else:
            logging.error("No relations or movements found in utterance.")

        if type_ == 'particle':
            self.detect_particles(particles)
        elif type_ == 'GMM':
            self.detect_GMM(GMM)
        else:
            logging.error('Wrong detection model specified.')

    def parse_utterance(self):
        """ Parse the input string into workable values.
        """

        for str_ in self.certainties:
            if str_ in self.utterance:
                self.certainty = str_
                break
        else:
            self.certainty = ''

        for str_ in self.targets:
            if str_ in self.utterance:
                self.target_name = str_
                break
        else:
            self.target_name = ''

        for str_ in self.relations:
            if str_ in self.utterance:
                self.relation = str_
                break
        else:
            self.relation = ''

        for str_ in self.groundings:
            if str_ in self.utterance:
                self.grounding_name = str_
                if str_ in self.groundings['object'].keys():
                    self.grounding = self.groundings['object'][str_]
                elif str_ in self.groundings['area'].keys():
                    self.grounding = self.groundings['area'][str_]
                break
        else:
            self.grounding_name = ''

        for str_ in self.movement_type:
            if str_ in self.utterance:
                self.movement_type = str_
                break
        else:
            self.movement_type = ''

        for str_ in self.movement_qualities:
            if str_ in self.utterance:
                self.movement_quality = str_
                break
        else:
            self.movement_quality = ''

        if self.utterance == '':
            utterance_is_well_formed = False
        else:
            utterance_is_well_formed = True
        return utterance_is_well_formed

    def translate_relation(self, relation_type='intrinsic'):
        """Translate the uttered relation to a likelihood label.
        """
        if relation_type == 'intrinsic':
            # Translate relation to zone label
            if self.relation == 'behind':
                translated_relation = 'back'
            elif self.relation == 'in front of':
                translated_relation = 'front'
            elif self.relation == 'left of':
                translated_relation = 'left'
            elif self.relation == 'right of':
                translated_relation = 'right'
            elif self.relation in ['inside', 'near', 'outside']:
                translated_relation = relation
            self.relation = translated_relation
        elif relation_type == 'relative':
            pass # <>TODO: Implement relative relations

    def translate_movement(self):
        """Translate the uttered movement to a likelihood label.

        """
        # Translate movement to motion model
        if self.movement_type == 'stopped':
            translated_movement = 'stopped'
        elif self.movement_quality == 'slowly':
            translated_movement = 'moving slowly'
        elif self.movement_quality == 'moderately':
            translated_movement = 'moving moderately'
        elif self.movement_quality == 'quickly':
            translated_movement = 'moving quickly'
        self.movement = translated_movement

    def detect_particles(self, particles):
        """Update particles based on sensor model.

        Parameters
        ----------
        particles : array_like
            The particle list, assuming [p,x,y,x_dot,y_dot], where x and y are
            position data and p is the particle's associated probability.

        """
        if self.detection_type == 'position':
            label = self.relation
        elif self.detection_type == 'movement':
            label = self.movement

        # <>TODO: include certainty
        for i, particle in enumerate(particles):
            state = particle[1:]

            if self.target_name == 'nothing' and self.positivity == 'not':
                self.target_name = 'a robot'
            elif self.target_name == 'nothing' or self.positivity == 'not':
                label = 'not ' + label
            particle[0] *= self.grounding.prob_at_state(state, label)

        # Renormalize
        particles[:, 0] /= sum(particles[:, 0])

        def detect_GMM(self, GMM):
            pass
