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

    # templates = {'spatial relation': ['certainty',
    #                                        'target',
    #                                        'positivity',
    #                                        ['spatial_relation:object', 'grounding:object'],
    #                                        ['spatial_relation:area', 'grounding:area'],
    #                                        ],
    #                   'action': ['certainty',
    #                              'target',
    #                              'positivity',
    #                              'action',
    #                              [''],
    #                              ['modifier'],
    #                              ['spatial_relation:movement', 'grounding:area'],
    #                              ['spatial_relation:movement', 'grounding:object'],
    #                              ],
    #                  }
    template_trees = {'spatial relation': [['certainty', 'target', 'positivity',
                                            'spatial_relation:object', 'grounding:object'],
                                           ['certainty', 'target', 'positivity',
                                            'spatial_relation:area', 'grounding:area'],
                                           ],
                      'action': [['certainty', 'target', 'positivity', 'action', ''],
                                 ['certainty', 'target', 'positivity', 'action', 'modifier'],
                                 ['certainty', 'target', 'positivity', 'action',
                                  'spatial_relation:movement', 'grounding:area'],
                                 ['certainty', 'target', 'positivity', 'action',
                                  'spatial_relation:movement', 'grounding:object'],
                                 ],
                      }
    template_object_mapping = {'spatial relation': SpatialRelationStatement,
                               'action': ActionStatement,
                               }

    def __init__(self, map_=None, add_more_relations=False, add_more_targets=False,
                 add_actions=False, add_certainties=False):
        self.components = {}
        self.templates = {'spatial relation': SpatialRelationStatement}

        # Define base template components
        self.components['certainty'] = ['know']
        self.components['positivity'] = ['is not', 'is']  # <>TODO: oh god wtf why does order matter
        self.components['spatial_relation'] = {'object': ['near'],
                                                'area': ['inside'],
                                               }
        self.components['target'] = ['nothing', 'a robot']

        # Define map-dependent components
        if map_ is None:
            map_ = Map(map_name='fleming')
        self.map = map_
        self.generate_groundings_from_map()
        if add_more_targets:
            self.generate_targets_from_map()

        # Define more spatial relation components
        if add_more_relations:
            self.components['spatial_relation']['object'] += ['behind',
                                                               'in front of',
                                                               'left of',
                                                               'right of',
                                                               ]
            self.components['spatial_relation']['area'] += ['near','outside']

        # Define action components
        if add_actions:
            self.components['action'] = ['stopped','moving']
            self.components['modifier'] = ['slowly', 'moderately', 'quickly']
            self.components['spatial_relation']['movement'] = ['around', 'toward']
            self.templates['action'] = ActionStatement
        else:
            self.template_trees = copy.deepcopy(StatementTemplate.template_trees)
            del self.template_trees['action']

        # Define certainty components
        if add_certainties:
            self.components['certainty'] += ['think']

        self._create_tree_structure()

    def generate_groundings_from_map(self):
        groundings = {}
        groundings['area'] = []
        for area_name in self.map.areas.keys():
            groundings['area'].append(add_article(area_name.lower()))
        # groundings['null'] = {}

        groundings['object'] = []

        # Add deckard (and other cops)
        if len(self.map.cops) == 0:
            groundings['object'].append("Deckard")
        else:
            for cop_name, cop in self.map.cops.iteritems():
                # if cop.has_relations:
                groundings['object'].append(cop.name)

        for object_name, obj in self.map.objects.iteritems():
            if obj.has_relations:
                grounding_name = add_article(obj.name.lower())
                groundings['object'].append(grounding_name)

        self.components['grounding'] = groundings

    def generate_targets_from_map(self):
        if len(self.map.robbers) == 0:
            additional_targets = ['Roy', 'Pris', 'Zhora']
        else:
            additional_targets = self.map.robbers.keys()

        self.components['target'] += additional_targets

    def generate_statements(self, autogenerate_softmax=False):
        """Generates a flat list of statement objects for each template.

        At last timing, 17.6 microseconds per statement, 2200 statements.
        """
        self.template_statements = {}

        # Look through templates (i.e. 'spatial relation', 'action', ...)
        for template_name, _ in self.templates.iteritems():

            # Get statement arguments from each node in the tree
            statement_args = []
            self.trees[template_name].get_statement_args(statement_args)

            # Generate statement objects for each statement argument set
            statements = []
            for args in statement_args:
                statement_class = self.template_object_mapping[template_name]
                statement = statement_class(autogenerate_softmax=autogenerate_softmax,
                                            map_=self.map,
                                            **args)
                statements.append(statement)
            self.template_statements[template_name] = statements

        self._prune_statements()

    def print_statements(self):
        for template_name, statements in self.template_statements.iteritems():
            str_ = '\n' + template_name.upper() + 's (' + str(len(statements)) + ')'
            print str_ + '\n' + '-' * len(str_)

            for s in statements:
                try:
                    print s
                    print s.softmax
                except AttributeError:
                    pass

    def _prune_statements(self):
        for template_name, statements in self.template_statements.iteritems():

            # Remove double negatives
            statements = [s for s in statements if not (s.target == 'nothing' and
                                                        s.positivity == 'is not')]

            # Remove 'stopped' with modifiers and groundings
            new_statements = []
            for s in statements:
                if not (hasattr(s, 'action') and s.action =='stopped'
                        and any((hasattr(s, 'modifier'),
                                 hasattr(s, 'grounding'),
                                 hasattr(s, 'spatial_relation')))):
                    new_statements.append(s)

            statements = new_statements

            self.template_statements[template_name] = statements

    def _get_node(self, tree, node_name):
        if tree is None:
            return None
        else:
            return tree.get_node(node_name, None)


    def _create_tree_structure(self):
        """Generates a tree form of all possible human sensor statements.

        Requires unique node names
        """
        self.trees = {}
        for template_name, templates in self.template_trees.iteritems():

            tree = None
            for template in templates:
                for depth, component_name in enumerate(template):

                    # Create new node if none exists in the tree
                    if self._get_node(tree, component_name) is None:
                        component_list = self._get_components(component_name)
                        node = Node(component_name, component_list)

                        # Add node to parent's children
                        if depth > 0:
                            parent_node = self._get_node(tree, template[depth - 1])
                            node.parent = parent_node
                            parent_node.children += [node]
                        else:
                            tree = node

            self.trees[template_name] = tree
            # print tree

    def _get_components(self, component_name):
        """Look up template components associated with a component name.

        e.g. 'actions' maps to ['stopped', 'moving']
        """
        if component_name == '':
            return list()
        component_names = component_name.split(':')

        split_names = []
        for name in component_names:
            split_names.append(name)

        component_list = self.components
        try:
            for key in split_names:
                component_list = component_list[key]
        except (KeyError):
            logging.error("No entry for {}".format(key))
        except (TypeError):
            pass

        return component_list

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
    def __init__(self, name, values, children=None, is_subtree_leaf=False):
        self.name = name
        self.values = values
        self.is_subtree_leaf = is_subtree_leaf
        if children is None:
            self.children = []

    def __repr__(self, level=0):
        str_ = "\t" * level + self.name + ": " + repr(self.values) + "\n"
        for child in self.children:
            str_ += child.__repr__(level + 1)
        return str_

    def get_node(self, node_name, node=None):

        if self.name == node_name:
            node = self
        else:
            for child in self.children:
                node = child.get_node(node_name, node)
        return node

    def get_statement_args(self, master_list, path_args=None):
        """Appends 
        """
        if path_args is None:
            path_args = {}
        path_args = path_args.copy()

        # Use the node's name as the parameter
        param = self.name
        if param.find(':') > -1:
            param = param.split(':')[0]

        # Append arguments of empty nodes
        if self.name == '':
            master_list.append(path_args.copy())

        # Use the node's values as arguments, branch for each child
        for arg in self.values:
            path_args[param] = arg

            # Append arguments of children
            for child in self.children:
                child.get_statement_args(master_list, path_args)

            # Append arguments of each path to master list at leaf nodes
            if len(self.children) == 0:
                master_list.append(path_args.copy())



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    # st = StatementTemplate(add_actions=True)
    st = StatementTemplate(add_more_relations=True, add_more_targets=True,
                           add_actions=True, add_certainties=True)
    st.generate_statements(autogenerate_softmax=False)
    st.print_statements()
    # print st.enumerate_combinations()