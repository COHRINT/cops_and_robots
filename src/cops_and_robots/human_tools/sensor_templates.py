#!/usr/bin/env python
"""
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

def generate_human_language_template(use_fleming=True, default_targets=True):
    """Generates speech templates

    Note: this function can be used to add attributes to a class instance with
    the following command (where 'self' is the class instance):

    self.__dict__.update(generate_human_language_template().__dict__)

    """
    if use_fleming:
        map_ = Map(map_name='fleming')

    # certainties = ['think', 'know']
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