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
from collections import namedtuple

from shapely.geometry import Point

from cops_and_robots.fusion.sensor import Sensor
from cops_and_robots.fusion.softmax import Softmax, speed_model, binary_speed_model
from cops_and_robots.fusion.gaussian_mixture import GaussianMixture
from cops_and_robots.fusion.variational_bayes import VariationalBayes
from cops_and_robots.map_tools.map import Map
from cops_and_robots.human_tools.nlp.chatter import Chatter

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

        super(Human, self).__init__(self.update_rate,
                                    self.has_physical_dimensions)

        # # self.certainties = ['think', 'know']
        # self.certainties = ['know']
        # self.positivities = ['is not', 'is']  # <>TODO: oh god wtf why does order matter
        # self.relations = {'object': ['behind',
        #                              'in front of',
        #                              'left of',
        #                              'right of',
        #                              'near',
        #                              ],
        #                   'area': ['inside',
        #                            'near',
        #                            'outside'
        #                            ]}
        # self.movement_types = ['moving', 'stopped']
        # self.movement_qualities = ['slowly', 'moderately', 'quickly']

        # self.groundings = {}
        # self.groundings['area'] = map_.areas

        # self.groundings['object'] = {}
        # for cop_name, cop in map_.cops.iteritems():
        #     # if cop.has_relations:
        #     self.groundings['object'][cop_name] = cop
        # for object_name, obj in map_.objects.iteritems():
        #     if obj.has_relations:
        #         self.groundings['object'][object_name] = obj

        # self.target_names = ['nothing', 'a robot'] + map_.robbers.keys()

        # Add all templates to this class
        self.__dict__.update(generate_human_language_template().__dict__)

        self.utterance = ''
        self.new_update = False

        # Set up the VB fusion parameters
        self.vb = VariationalBayes()

        if web_interface_topic != 'python':
            # Subscribe to web interface
            import rospy
            from std_msgs.msg import String
            rospy.Subscriber(web_interface_topic, String, self.callback)

        self.chatter = Chatter()

    def callback(self, msg):
        logging.info('I heard {}'.format(msg.data))
        self.utterance = msg.data
        self.new_update = True

    def get_measurement(self):
        """Update a fusion engine's probability from human sensor updates.

        """

        try:
            well_formed = self.parse_utterance()
        except:
            logging.error("Can't parse \"{}\"".format(utterance))

        if not well_formed:
            logging.debug('No utterance to parse!')
            return

        if self.relation != '':
            self.translate_relation()
            self.detection_type = 'position'
        elif self.movement_type != '':
            self.translate_movement()
            self.detection_type = 'movement'
        else:
            logging.error("No relations or movements found in utterance.")

        # Swap left-and-right for Deckard, as those are ego-centric
        if self.grounding.name.lower() == 'deckard':
            if self.relation == 'Right':
                self.relation = 'Left'
            elif self.relation == 'Left':
                self.relation = 'Right'

        # Define relations lazily, or for dynamic targets
        if not hasattr(self.grounding, 'relations') \
            or self.grounding.name.lower() == 'deckard':
            logging.info("Defining relations because {} didn't have any."
                         .format(self.grounding.name))
            self.grounding.define_relations()

        # Position update
        relation_label = self.relation
        if self.target_name == 'nothing' and self.positivity == 'is not':
            relation_label = relation_label.replace('Not ', '')
        elif self.target_name == 'nothing' or self.positivity == 'is not':
            relation_label = 'Not ' + relation_label

        parsed_measurement = {'grounding': self.grounding,
                              'relation': relation_label,
                              'relation class': self.relation,
                              'target': self.target_name
                              }
        return parsed_measurement

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

                # #<>TODO: REMOVE THIS HORRIBLE HACK!
                # if str_ == 'Roy':
                #     str_ = 'Pris'
                # elif str_ == 'Pris':
                #     str_ = 'Roy'


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
                if str_ in self.utterance.lower():
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

        # for str_ in self.movement_types:
        #     if str_ in self.utterance:
        #         self.movement_type = str_

        #         logging.info(str_)

        #         break
        # else:
        #     self.movement_type = ''

        # for str_ in self.movement_qualities:
        #     if str_ in self.utterance:
        #         self.movement_quality = str_

        #         logging.info(str_)

        #         break
        # else:
        #     self.movement_quality = ''

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


def generate_human_language_template(use_fleming=True, default_targets=True):
    """Generates speech templates

    Note: this function can be used to add attributes to a class instance with
    the following command (where 'self' is the class instance):

    self.__dict__.update(generate_human_language_template().__dict__)

    """
    if use_fleming:
        map_ = Map(map_name='fleming')

    # self.certainties = ['think', 'know']
    certainties = ['know']
    positivities = ['is not', 'is']  # <>TODO: oh god wtf why does order matter
    relations = {'object': ['behind',
                                 'in front of',
                                 'left of',
                                 'right of',
                                 'near',
                                 ],
                  'area': ['inside',
                           'near',
                           'outside'
                           ]}
    actions = ['moving', 'stopped']
    modifiers = ['slowly', 'moderately', 'quickly', 'around', 'toward']

    movement_types = ['moving', 'stopped']
    movement_qualities = ['slowly', 'moderately', 'quickly']

    groundings = {}
    groundings['area'] = map_.areas
    groundings['null'] = {}

    groundings['object'] = {}
    for cop_name, cop in map_.cops.iteritems():
        # if cop.has_relations:
        groundings['object'][cop_name] = cop
    for object_name, obj in map_.objects.iteritems():
        if obj.has_relations:
            groundings['object'][object_name] = obj

    if default_targets:
        target_names = ['nothing', 'a robot', 'Roy','Pris','Zhora']
    else:
        target_names = ['nothing', 'a robot'] + map_.robbers.keys()

    Template = namedtuple('Template', 'certainties positivities relations' 
        + ' actions modifiers groundings target_names')
    Template.__new__.__defaults__ = (None,) * len(Template._fields)

    template = Template(certainties,
                        positivities,
                        relations,
                        actions,
                        modifiers,
                        groundings,
                        target_names
                        )

    return template

if __name__ == '__main__':
    pass