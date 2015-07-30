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

from cops_and_robots.fusion.sensor import Sensor
from cops_and_robots.fusion.softmax import Softmax, speed_model, binary_speed_model
from cops_and_robots.fusion.gaussian_mixture import GaussianMixture
from cops_and_robots.fusion.variational_bayes import VariationalBayes


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
    def __init__(self, map_=None, web_interface_topic='python',
                 false_alarm_prob=0.2):
        self.update_rate = None
        self.has_physical_dimensions = False
        self.speed_model = speed_model()
        self.false_alarm_prob = false_alarm_prob
        # self.speed_model = binary_speed_model()

        super(Human, self).__init__(self.update_rate,
                                    self.has_physical_dimensions)

        # self.certainties = ['think', 'know']
        self.certainties = ['know']
        self.positivities = ['is not', 'is']  # <>TODO: oh god wtf why does order matter
        self.relations = {'object': ['behind',
                                     'in front of',
                                     'left of',
                                     'right of',
                                     ],
                          'area': ['inside',
                                   'near',
                                   'outside'
                                   ]}
        self.movement_types = ['moving', 'stopped']
        self.movement_qualities = ['slowly', 'moderately', 'quickly']

        self.groundings = {}
        self.groundings['area'] = map_.areas

        self.groundings['object'] = {}
        for cop_name, cop in map_.cops.iteritems():
            if cop.has_relations:
                self.groundings['object'][cop_name] = cop
        for object_name, obj in map_.objects.iteritems():
            if obj.has_relations:
                self.groundings['object'][object_name] = obj

        self.target_names = ['nothing', 'a robot'] + map_.robbers.keys()
        self.utterance = ''
        self.new_update = False

        # Set up the VB fusion parameters
        self.vb = VariationalBayes()

        if web_interface_topic != 'python':
            # Subscribe to web interface
            print web_interface_topic
            import rospy
            from std_msgs.msg import String
            rospy.Subscriber(web_interface_topic, String, self.callback)

    def callback(self, msg):
        logging.debug('Processing sensor')
        self.utterance = msg.data
        self.new_update = True

    def detect(self, filter_name, type_="particle", particles=None, prior=None):
        """Update a fusion engine's probability from human sensor updates.

        Parameters
        ----------
        particles : array_like
            The particle list, assuming [x,y,p], where x and y are position
            data and p is the particle's associated probability. `None` if not
            using particles.
        """

        if not self.parse_utterance():
            logging.debug('No utterance to parse!')
            return

        # End detect loop if not the right target
        if self.target_name not in ['nothing', 'a robot', filter_name]:
            logging.debug('Target {} is not in {} Looking for {}.'
                .format(filter_name, self.utterance, self.target_name))
            return

        if self.relation != '':
            self.translate_relation()
            self.detection_type = 'position'
        elif self.movement_type != '':
            self.translate_movement()
            self.detection_type = 'movement'
        else:
            logging.error("No relations or movements found in utterance.")

        if type_ == 'particle':
            self.detect_particles(particles)
            return True
        elif type_ == 'gauss sum':
            return self.detect_probability(prior)
        else:
            logging.error('Wrong detection model specified.')

    def parse_utterance(self):
        """ Parse the input string into workable values.
        """
        logging.debug('Parsing: {}'.format(self.utterance))
        logging.debug('Groundings: {}'.format(self.groundings))

        for str_ in self.certainties:
            if str_ in self.utterance:
                self.certainty = str_
                break
        else:
            self.certainty = ''

        for str_ in self.target_names:
            if str_ in self.utterance:
                self.target_name = str_
                break
        else:
            self.target_name = ''

        for str_ in self.positivities:
            if str_ in self.utterance:
                self.positivity = str_
                break
        else:
            self.positivity = ''

        for str_type in self.relations:
            for str_ in self.relations[str_type]:
                if str_ in self.utterance:
                    self.relation = str_
                    break
            else:
                continue
            break
        else:
            self.relation = ''

        for str_type in self.groundings:
            for str_ in self.groundings[str_type].keys():
                str_ = str_.lower()
                if str_ in self.utterance:
                    str_ = str_.title()
                    self.grounding = self.groundings[str_type][str_]
                    break
            else:
                continue
            break
        else:
            self.grounding = None

        logging.debug('Utterance: {}'.format(self.utterance))
        logging.debug('Grounding: {}'.format(self.grounding))

        for str_ in self.movement_types:
            if str_ in self.utterance:
                self.movement_type = str_

                logging.info(str_)

                break
        else:
            self.movement_type = ''

        for str_ in self.movement_qualities:
            if str_ in self.utterance:
                self.movement_quality = str_

                logging.info(str_)

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
                translated_relation = 'Back'
            elif self.relation == 'in front of':
                translated_relation = 'Front'
            elif self.relation == 'left of':
                translated_relation = 'Left'
            elif self.relation == 'right of':
                translated_relation = 'Right'
            elif self.relation in ['inside', 'near', 'outside']:
                translated_relation = self.relation.title()
            self.relation = translated_relation
        elif relation_type == 'relative':
            pass  # <>TODO: Implement relative relations

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

        if not hasattr(self.grounding, 'relations'):
            self.grounding.define_relations()

        # <>TODO: include certainty
        if self.detection_type == 'position':
            label = self.relation
            if self.target_name == 'nothing' and self.positivity == 'not':
                label = 'a robot'
            elif self.target_name == 'nothing' or self.positivity == 'not':
                label = 'Not ' + label

            for i, particle in enumerate(particles):
                state = particle[1:3]
                particle[0] *= self.grounding.relations.probability(state=state, class_=label)

        elif self.detection_type == 'movement':
            label = self.movement
            if self.target_name == 'nothing' and self.positivity == 'not':
                label = 'a robot'
            elif self.target_name == 'nothing' or self.positivity == 'not':
                label = 'Not ' + label

            for i, particle in enumerate(particles):
                state = np.sqrt(particle[3] ** 2 + particle[4] ** 2)
                particle[0] *= self.speed_model.probability(state=state, class_=label)

        # Renormalize
        particles[:, 0] /= sum(particles[:, 0])

    def detect_probability(self, prior):
        if not hasattr(self.grounding,'relations'):
            self.grounding.define_relations()

        # Position update
        label = self.relation
        if self.target_name == 'nothing' and self.positivity == 'is not':
            label = 'a robot'
        elif self.target_name == 'nothing' or self.positivity == 'is not':
            label = 'Not ' + label

        likelihood = self.grounding.relations.binary_models[self.relation]
        mu, sigma, beta = self.vb.update(measurement=label,
                                         likelihood=likelihood,
                                         prior=prior,
                                         use_LWIS=False,
                                         )

        gm = GaussianMixture(beta, mu, sigma)
        alpha = self.false_alarm_prob / 2
        posterior = prior.combine_gms(gm, alpha)
        # Weight based on human possibly being wrong
        return posterior
