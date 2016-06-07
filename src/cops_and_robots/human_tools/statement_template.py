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
    default_components = {'certainty': 'know',
                          'positivity': 'is',
                          'target': 'a robot',
                          }
    statement_classes = {'spatial relation': SpatialRelationStatement,
                         'action': ActionStatement,
                         }

    def __init__(self, map_=None, add_more_relations=False, add_more_targets=False,
                 add_actions=False, add_certainties=False):
        self.components = {}
        self.templates = {'spatial relation': SpatialRelationStatement}

        # Define base template components
        self.components['certainty'] = ['know']
        self.components['positivity'] = ['is not', 'is']  # <>TODO: Check if order matters
        self.components['spatial_relation'] = {'object': ['near'],
                                                'area': ['inside'],
                                               }
        self.components['target'] = ['nothing', 'a robot']

        # Define map-dependent components
        if map_ is None:
            map_ = Map(map_name='fleming')
        self.map = map_
        self._generate_groundings_from_map()
        if add_more_targets:
            self._generate_targets_from_map()

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
        return self.template_statements

    def print_statements(self):
        for template_name, statements in self.template_statements.iteritems():

            str_ = '\n' + template_name.upper() + 's (' + str(len(statements)) + ')'
            print str_ + '\n' + '-' * len(str_)

            for s in statements:
                print s

    def _generate_groundings_from_map(self):
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

    def _generate_targets_from_map(self):
        if len(self.map.robbers) == 0:
            additional_targets = ['Roy', 'Pris', 'Zhora']
        else:
            additional_targets = self.map.robbers.keys()

        self.components['target'] += additional_targets

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


    def _create_tree_structure(self):
        """Generates a tree form of all possible human sensor statements.

        Requires unique node names.
        """
        self.trees = {}
        for template_name, templates in self.template_trees.iteritems():

            tree = None
            for template in templates:
                for depth, component_name in enumerate(template):

                    # Create new node if none exists in the tree
                    if tree is None or tree.get_node(component_name) is None:

                        # Create node
                        component_list = self._get_components(component_name)
                        node = Node(component_name, component_list)

                        # Add node to parent's children
                        if depth > 0:
                            parent_node = tree.get_node(template[depth - 1])
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
    def __init__(self, name, values, children=None):
        self.name = name
        self.values = values
        if children is None:
            self.children = []

    def __repr__(self, level=0):
        str_ = "\t" * level + self.name + ": " + repr(self.values) + "\n"
        for child in self.children:
            str_ += child.__repr__(level + 1)
        return str_

    def get_node(self, node_name, node=None):
        """Searches the tree for a specific node by name.
        """
        if self.name == node_name:
            node = self
        else:
            for child in self.children:
                node = child.get_node(node_name, node)
        return node

    def get_statement_args(self, master_list, path_args=None):
        """Appends path of arg values through tree to `master_list`.
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

def get_all_statements(autogenerate_softmax=False, flatten=False,
                       *args, **kwargs):
    st = StatementTemplate(*args, **kwargs)
    st.generate_statements(autogenerate_softmax=autogenerate_softmax)
    statements = st.template_statements

    if flatten:
        statements = []
        for _, ts in st.template_statements.iteritems():
            statements += statements + ts
    return statements


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    # st = StatementTemplate(add_more_relations=True, add_more_targets=True, add_actions=True, add_certainties=True)
    st = StatementTemplate()
    st.generate_statements(autogenerate_softmax=False)
    st.print_statements()
    # print st.enumerate_combinations()