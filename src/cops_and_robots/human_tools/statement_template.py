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
import copy

from cops_and_robots.map_tools.map import Map
from cops_and_robots.human_tools.statement import (ActionStatement,
                                                   SpatialRelationStatement,
                                                   add_article,
                                                   )


class StatementTemplate(object):
    """Defines the possible components of a `Statement`.

    Each `Statement` object is tied to one unique `Softmax` object. This is an
    *n-to-one* relationship.

    Parameters
    ----------
    add_more_relations : bool, optional
        Add additional sensor statements such as 'left of', 'right of', etc.
        Default is `False`.
    add_actions : bool, optional
        Add additional sensor statements for movement actions. Default is 
        `False`.
    add_certainties : bool, optional
        Add additional sensor statements for 'I think'. Default is `False`.
    """

    tree_templates = {'spatial relation': ['certainty',
                                           'target',
                                           'positivity',
                                           ['spatial_relation:object', 'grounding:object'],
                                           ['spatial_relation:area', 'grounding:area'],
                                           ],
                      'action': ['certainty',
                                 'target',
                                 'positivity',
                                 'action',
                                 ['modifier'],
                                 ['spatial_relation:movement', 'grounding:area'],
                                 ['spatial_relation:movement', 'grounding:object'],
                                 ],
                     }
    plural_mapping = {'target': 'targets',
                      'certainty': 'certainties',
                      'positivity': 'positivities',
                      'spatial_relation': 'spatial_relations',
                      'grounding': 'groundings',
                      'object': 'objects',
                      'movement': 'movements',
                      'area': 'areas',
                      'action': 'actions',
                      'modifier': 'modifiers',
                      }
    template_object_mapping = {'spatial relation': SpatialRelationStatement,
                               'action': ActionStatement,
                               }

    def __init__(self, map_=None, add_more_relations=False,
                 add_actions=False, add_certainties=False):
        self.components = {}
        self.templates = {'spatial relation': SpatialRelationStatement}

        # Define base template components
        self.components['certainties'] = ['know']
        self.components['positivities'] = ['is not', 'is']  # <>TODO: oh god wtf why does order matter
        self.components['spatial_relations'] = {'objects': ['near'],
                                                'areas': ['inside'],
                                               }
        self.components['targets'] = ['nothing', 'a robot']

        # Define map-dependent components
        if map_ is None:
            map_ = Map(map_name='fleming')
        self.map = map_
        self.generate_groundings_from_map()
        self.generate_targets_from_map()

        # Define more spatial relation components
        if add_more_relations:
            self.components['spatial_relations']['objects'] += ['behind',
                                                               'in front of',
                                                               'left of',
                                                               'right of',
                                                               ]
            self.components['spatial_relations']['areas'] += ['near','outside']

        # Define action components
        if add_actions:
            self.components['actions'] = ['stopped','moving']
            self.components['modifiers'] = ['slowly', 'moderately', 'quickly']
            self.components['spatial_relations']['movements'] = ['around', 'toward']
            self.templates['action'] = ActionStatement
        else:
            self.tree_templates = copy.deepcopy(StatementTemplate.tree_templates)
            del self.tree_templates['action']

        # Define certainty components
        if add_certainties:
            self.components['certainties'] += ['think']

        self.create_tree_structure()

    def generate_groundings_from_map(self):
        groundings = {}
        groundings['areas'] = []
        for area_name in self.map.areas.keys():
            groundings['areas'].append(add_article(area_name.lower()))
        # groundings['null'] = {}

        groundings['objects'] = []

        # Add deckard (and other cops)
        if len(self.map.cops) == 0:
            groundings['objects'].append("Deckard")
        else:
            for cop_name, cop in self.map.cops.iteritems():
                # if cop.has_relations:
                groundings['objects'].append(cop.name)


        for object_name, obj in self.map.objects.iteritems():
            if obj.has_relations:
                grounding_name = add_article(obj.name.lower())
                groundings['objects'].append(grounding_name)

        # groundings['objects'] = {}
        # for cop_name, cop in map_.cops.iteritems():
        #     # if cop.has_relations:
        #     groundings['objects'][cop_name] = cop
        # for object_name, obj in map_.objects.iteritems():
        #     if obj.has_relations:
        #         groundings['objects'][object_name] = obj

        self.components['groundings'] = groundings

    def generate_targets_from_map(self):
        if len(self.map.robbers) == 0:
            additional_targets = ['Roy', 'Pris', 'Zhora']
        else:
            additional_targets = self.map.robbers.keys()

        self.components['targets'] += additional_targets

    def enumerate_combinations(self, verbose=False):
        n_per_template = {key: 1 for key, _ in self.templates.iteritems()}

        # Look through templates (i.e. 'spatial relation', 'action', ...)
        for template_name, tree in self.trees.iteritems():

            n = self.trees[template_name].enumerate_subtree_components()
            n_per_template[template_name] = n

        if verbose:
            for template_name, n in n_per_template.iteritems():

                print '\n' + template_name.upper() + '\n' + '-' * len(template_name)
                print '{} possible combinations'.format(n)

        return n_per_template

    def generate_sentences(self, verbose=False):
        """Generates all possible strings for each template.
        """
        sentences_per_template = {}

        # Look through templates (i.e. 'spatial relation', 'action', ...)
        for template_name, tree in self.trees.iteritems():

            strings = []
            self.trees[template_name].add_to_string("I", strings)

            sentences_per_template[template_name] = strings

        if verbose:
            for template_name, sentences in sentences_per_template.iteritems():
                print '\n' + template_name.upper() + '\n' + '-' * len(template_name)
                for s in sentences:
                    print s

        return sentences_per_template

    def generate_statements(self, autogenerate_softmax=False, verbose=False):
        """Generates all possible statement objects for each template.

        At last timing, 17.6 microseconds per statement, 2200 statements.
        """
        statements_per_template = {}

        # Look through templates (i.e. 'spatial relation', 'action', ...)
        for template_name, tree in self.trees.iteritems():

            statement_args = []
            self.trees[template_name].get_statement_args(None, statement_args)

            statements = []
            for args in statement_args:
                statement_class = self.template_object_mapping[template_name]
                statement = statement_class(autogenerate_softmax=autogenerate_softmax,
                                            map_=self.map,
                                            **args)
                statements.append(statement)
            statements_per_template[template_name] = statements

            if verbose:
                for template_name, statements in statements_per_template.iteritems():
                    print '\n' + template_name.upper() + '\n' + '-' * len(template_name)
                    for s in statements:
                        try:
                            print s
                            print s.softmax
                        except AttributeError:
                            pass

        return statements_per_template

    def prune_edge_statements():

    def create_tree_structure(self):
        self.trees = {}
        for template_name, tree_template in self.tree_templates.iteritems():
            self.trees[template_name] = None

            subtree_parent = None
            for depth, component_name in enumerate(tree_template):

                # Deal with any component name list as a subtree branch
                if isinstance(component_name, list):
                    # Define subtree parent if not defined
                    if subtree_parent is None:
                        subtree_parent = parent_node

                    # Go through subtrees defined by component name lists
                    for subtree_depth, c in enumerate(component_name):
                        name, component_list = self._get_components(c)
                        node = Node(name, component_list)
                        # nodes.append(node)

                        # Set first node as subtree root element
                        if subtree_depth == 0:
                            parent_node = subtree_parent

                        # Add children and update parent
                        parent_node.children += [node]
                        parent_node = node
                else:
                    # Create node
                    name, component_list = self._get_components(component_name)
                    node = Node(name, component_list)

                    # Set first node as root element, assign other nodes to parents
                    if depth == 0:
                        self.trees[template_name] = node
                    else:
                        parent_node.children += [node]
                    parent_node = node


    def print_trees(self):
        for template_name, _ in self.trees.iteritems():
            print template_name.upper() + '\n' + '-' * len(template_name)
            print self.trees[template_name]

    def _get_components(self, component_name):
        component_names = component_name.split(':')

        split_names = []
        for name in component_names:
            nicename = StatementTemplate.plural_mapping[name]
            split_names.append(nicename)

        component_list = self.components
        try:
            for key in split_names:
                component_list = component_list[key]
        except (KeyError):
            logging.error("No entry for {}".format(key))
        except (TypeError):
            pass

        return component_name, component_list

        # {'spatial relation': ['certainty',
        #                                    'target',
        #                                    'positivity',
        #                                    ['spatial_relation:object', 'grounding:object'],
        #                                    ['spatial_relation:area', 'grounding:area'],
        #                                    ],
        #               'action': ['certainty',
        #                          'target',
        #                          'positivity',
        #                          'action',
        #                          ['modifier'],
        #                          ['spatial_relation:movement', 'grounding:area'],
        #                          ['spatial_relation:movement', 'grounding:object'],
        #                          ],
        #              }

def generate_sensor_statements(autogenerate_softmax=True, strings_only=False):
    """Creates all ``Statement`` objects, possibly precomputing softmax models.
    """


    certainties = self.human_sensor.certainties
    targets = self.target_order
    positivities = self.human_sensor.positivities
    groundings = self.human_sensor.groundings

    # Create all possible questions and precompute their likelihoods
    n_statements = statement_template.enumerate_combinations()
    statements = []

    i = 0
    #<>TODO: include certainties
    for grounding_type_name, grounding_type in groundings.iteritems():
        for grounding_name, grounding in grounding_type.iteritems():
            grounding_name = grounding_name.lower()
            if grounding_name == 'deckard':
                continue
            if grounding_name.find('the') == -1:
                grounding_name = 'the ' + grounding_name
            relation_names = grounding.relations.binary_models.keys()
            for relation_name in relation_names:
                relation = relation_name

                # Make relation names grammatically correct
                relation_name = relation_name.lower()
                if relation_name == 'front':
                    relation_name = 'in front of'
                elif relation_name == 'back':
                    relation_name = 'behind'
                elif relation_name == 'left':
                    relation_name = 'left of'
                elif relation_name == 'right':
                    relation_name = 'right of'

                # Ignore certain questons (incl. non-minimal)
                if grounding_type_name == 'object':
                    if relation_name in ['inside', 'outside']:
                        continue

                    if self.minimize_questions:
                        # Near only
                        if relation_name != 'near':
                            continue

                if grounding_type_name == 'area' \
                    and self.minimize_questions: 
                    
                    # Inside only
                    if relation_name != 'inside':
                        continue

                for target in targets:
                    # Write question
                    question_str = "Is " + target + " " + relation_name \
                        + " " + grounding_name +"?"
                    self.all_questions.append(question_str)

                    # Calculate likelihood
                    self.all_likelihoods[i]['question'] = question_str
                    self.all_likelihoods[i]['probability'] = \
                        grounding.relations.probability(class_=relation)
                    self.all_likelihoods[i]['time_last_answered'] = -1
                    i += 1
    logging.info('Generated {} questions.'.format(len(self.all_questions)))


def generate_human_language_template(use_fleming=True, default_targets=True):
    """Generates speech templates

    Note: this function can be used to add attributes to a class instance with
    the following command (where 'self' is the class instance):

    self.__dict__.update(generate_human_language_template().__dict__)

    """
    
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


class Node(object):
    """Element of Tree data structure."""
    def __init__(self, name, value, children=None):
        self.name = name
        self.value = value
        if children is None:
            self.children = []

    def __repr__(self, level=0):
        str_ = "\t" * level + self.name + ": " + repr(self.value) + "\n"
        for child in self.children:
            str_ += child.__repr__(level + 1)
        return str_

    def enumerate_subtree_components(self):
        n = len(self.value)

        # Return n at leaf nodes
        if len(self.children) == 0:
            return n

        # Find n for all children
        branch_n = 0
        for child in self.children:
            branch_n += child.enumerate_subtree_components()

        return n * branch_n

    def add_to_string(self, path_str, master_list):
        for str_ in self.value:
            str_ = path_str + ' ' + str_

            # Append strings of children
            root_str = str_
            for child in self.children:
                str_ = child.add_to_string(root_str, master_list)

            # Append a period and append to master list at leaf nodes
            if len(self.children) == 0:
                str_ += '.'
                master_list.append(str_)
        return str_

    def get_statement_args(self, path_args, master_list):
        if path_args is None:
            path_args = {}

        param = self.name
        if param.find(':') > -1:
            param = param.split(':')[0]

        for arg in self.value:
            path_args[param] = arg

            # Append strings of children
            for child in self.children:
                path_args = child.get_statement_args(path_args, master_list)

            # Append arguments of each path to master list at leaf nodes
            if len(self.children) == 0:
                # print path_args
                master_list.append(path_args.copy())
                # # del path_args
                # print id(master_list), master_list

        return path_args

[{'certainty': 'know', 
  'target': 'a robot',
  'positivity': 'is',
  'action': 'moving',
  'modifier': 'quickly',
 },
 {'certainty': 'know', 
  'target': 'a robot',
  'positivity': 'is',
  'action': 'moving',
  'modifier': 'slowly',
  }]


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    st = StatementTemplate()
    # st = StatementTemplate(add_more_relations=True, add_actions=True, add_certainties=True)
    # st.print_trees()
    st.enumerate_combinations(verbose=False)
    st.generate_sentences(verbose=False)
    st.generate_statements(autogenerate_softmax=True, verbose=True)
    # print st.enumerate_combinations()