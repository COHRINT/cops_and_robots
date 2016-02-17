#!/usr/bin/env python
from __future__ import division
"""MODULE_DESCRIPTION"""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import logging
import daft

from cops_and_robots.fusion.human import generate_human_language_template


class TDC(object):
    """Target description clause.

    A target description clause provides a formal structure for semantically
    tagged natural language.
    
    Parameters
    ----------
    type_ : str, optional
        The template type for the TDC.
    phrase : str, optional
        The untagged input phrase associated with the TDC.

    Attributes
    ----------
    templates : dict, class attr
        The list of tags that define a type of TDC, i.e. 'action' and 
        'spatial relation' have different sets of tags.
    type : str
        The template type.
    target : str_
        The target identified by the TDC. Always required, regardless of
        template, but inherited by parent TDC if missing. e.g. 'a robot'
    positivity : str_
        The positivity or negativity of the TDC statement (i.e. "is" vs.
        "is not". Always required, regardless of template, but inherited by
        parent TDC if missing.
    grounding : str_, optional
        A reference object captured in the TDC. Optional for 'action'
        templates, but not for 'spatial relation' templates. e.g. 'the desk'.
    spatial_relation : str_, optional
        The relation to the grounding captured by the TDC. Required for
        'spatial relation' templates, not included in 'action' templates.
        e.g. 'near'.
    action : str_, optional
        The movement of the target specified by the TDC. Required for 'action'
        templates, not included in 'spatial relation' templates. e.g. 'moving'.
    modifier : str_, optional
        A modifier to the action. Not included in 'spatial relation' templates,
        optional for 'action' templates. e.g. 'away from'.

    """
    templates = {'spatial relation': ['TARGET', 'POSITIVITY', 'GROUNDING',
                                      'SPATIALRELATION'],
                 'action': ['TARGET','POSITIVITY','ACTION','MODIFIER',
                            'GROUNDING'],
                }
    
    def __init__(self, type_='', phrase='', show_children=False):
        self.type = type_
        self.phrase = phrase
        self.show_children = show_children
        self.parsing = {}
        self.__target = ''
        self.__positivity = ''
        self.__grounding = ''
        self.__spatial_relation = ''
        self.__action = ''
        self.__modifier = ''
        self.child_TDCs = []
        if len(self.type) > 0:
            self.template = TDC.templates[self.type]
        else:
            self.template = []
    
    def __repr__(self):
        str_ = "{} TDC: {}".format(self.type.title(), self.parsing)
        if self.show_children:
            if len(self.child_TDCs) > 0:
                str_ += " with children: "
                for child_TDC in self.child_TDCs:
                    str_ += child_TDC.__repr__()
        return str_
    
    @property
    def target(self):
        return self.__target
    
    @target.setter
    def target(self, target):
        if self._check_input(target, 'TARGET'):
            self.parsing['TARGET'] = target
            self.__target = target

    @property
    def positivity(self):
        return self.__positivity
    
    @positivity.setter
    def positivity(self, positivity):
        if self._check_input(positivity, 'POSITIVITY'):
            self.parsing['POSITIVITY'] = positivity
            self.__positivity = positivity
    
    @property
    def grounding(self):
        return self.__grounding
    
    @grounding.setter
    def grounding(self, grounding):
        if self._check_input(grounding, 'GROUNDING'):
            self.parsing['GROUNDING'] = grounding
            self.__grounding = grounding 
        
    @property
    def spatial_relation(self):
        return self.__spatial_relation
    
    @spatial_relation.setter
    def spatial_relation(self, spatial_relation):
        if self._check_input(spatial_relation, 'SPATIALRELATION', 'spatial relation'):
            self.parsing['SPATIALRELATION'] = spatial_relation
            self.__spatial_relation = spatial_relation
        
    @property
    def action(self):
        return self.__action
    
    @action.setter
    def action(self, action):
        if self._check_input(action, 'ACTION', 'action'):
            self.parsing['ACTION'] = action
            self.__action = action
        
    @property
    def modifier(self):
        return self.__modifier
    
    @modifier.setter
    def modifier(self, modifier):
        if self._check_input(modifier, 'MODIFIER', 'action'):
            self.parsing['MODIFIER'] = modifier
            self.__modifier = modifier 
        
    def _check_input(self, input_span, true_span_tag, TDC_type=''):
        # Check for template matching
        is_conflicting = ((true_span_tag not in self.template) 
                          and len(self.template) > 0)
        
        # Check for duplicate spans
        current_span = self.get_spans_from_tags([true_span_tag])[0][0]
        has_duplicate = len(current_span) > 0
        
        # Add to/new child TDC if necessary
        if is_conflicting or has_duplicate:
            if len(self.child_TDCs) > 0:
                self.child_TDCs[0].add_span([input_span, true_span_tag])
            else:    
                new_tdc = TDC(type_=TDC_type, phrase=self.phrase)
                new_tdc.add_span([input_span, true_span_tag])
                logging.debug("Adding a child TDC to this {} --> {}"
                              .format(self, new_tdc))
                self.child_TDCs.append(new_tdc)
            return False
        
        # Set template and type
        if len(TDC_type) > 0:
            self.type = TDC_type
        if len(self.type) > 0:
            self.template = TDC.templates[self.type]
            
        return True
        
    def add_span(self, tagged_span=[], span='', span_tag=''):
        if len(tagged_span) > 0:
            span, span_tag = tagged_span
            
        if span_tag == 'NULL':
            logging.debug("Skipping null spans.")
            return
            
        if len(tagged_span) == 1 or len(tagged_span) > 2:
            logging.warning("You must specify a word and its semantic tag!")
            return
            
        if span_tag == 'TARGET':
            self.target = span
        elif span_tag == 'POSITIVITY':
            self.positivity = span
        elif span_tag == 'GROUNDING':
            self.grounding = span
        elif span_tag == 'SPATIALRELATION':
            self.spatial_relation = span
        elif span_tag == 'ACTION':
            self.action = span
        elif span_tag == 'MODIFIER':
            self.modifier = span
        
    def get_spans_from_tags(self, span_tag, include_empty_spans=True):
        if type(span_tag) is not type(list()):
            span_tag = [span_tag]
        parsed_spans = []
        
        for span_tag in span_tag:
            if span_tag == 'TARGET':
                span = self.target
            elif span_tag == 'POSITIVITY':
                span = self.positivity
            elif span_tag == 'GROUNDING':
                span = self.grounding
            elif span_tag == 'SPATIALRELATION':
                span = self.spatial_relation
            elif span_tag == 'ACTION':
                span = self.action
            elif span_tag == 'MODIFIER':
                span = self.modifier
                
            if not include_empty_spans and span == '':
                continue

            parsed_spans.append([span, span_tag])
        return parsed_spans

    def to_tagged_phrase(self):
        return [[t, w] for t,w in self.parsing]


class TDC_Collection(object):
    """A collection of TDCs

    """

    def __init__(self, tagged_document=None):
        self.TDCs = []
        if tagged_document is not None:
            self.parse_tagged_document(tagged_document)

        self.tag_order = {'TARGET': 0,
                          'POSITIVITY': 1,
                          'SPATIALRELATION': 2,
                          'ACTION': 3,
                          'MODIFIER': 4,
                          'GROUNDING': 5,
                          }
        
    def parse_tagged_document(self, tagged_document):
        phrases = self.split_phrases(tagged_document)
        self.get_TDCs_from_phrases(phrases)
        self.update_children_from_parents()
        self.flatten_TDCs()
        self.include_defaults()
        self.prune_TDCs()

    def split_phrases(self, tagged_document):
        """Identify individual phrases in a tagged document"""
        phrases = []
        phrase = []
        for tagged_phrase in tagged_document:
            if tagged_phrase[0] not in ['.', '!', '?']:
                phrase.append(tagged_phrase)
            else:
                phrases.append(phrase)
                phrase = []
        return phrases
        
    def get_TDCs_from_phrases(self, phrases):
        """Create TDCs for each phrase"""
        for phrase in phrases:
            phrase_str = " ".join([s[0] for s in phrase])
            tdc = TDC(phrase=phrase_str)
            for tagged_span in phrase:
                tdc.add_span(tagged_span)
            self.TDCs.append(tdc)

    def update_children_from_parents(self):
        """Have child TDCs inherit parent information"""
        for tdc in self.TDCs:
            for child_tdc in tdc.child_TDCs:
                
                child_tags = TDC.templates[child_tdc.type]
                parsed_child_spans = child_tdc.get_spans_from_tags(child_tags)
                
                for parsed_child_span in parsed_child_spans:
                    
                    # Replace null span with parent span
                    if len(parsed_child_span[0]) == 0:
                        parent_span = tdc.get_spans_from_tags(parsed_child_span[1])[0]
                        child_tdc.add_span([parent_span[0], parsed_child_span[1]])

    def flatten_TDCs(self, remove_children=False):
        """Flatten TDC list, optionally removing children"""
        child_TDCs = []
        for tdc in self.TDCs:
            for i, child_TDC in enumerate(tdc.child_TDCs):
                child_TDCs.append(child_TDC)
                if remove_children:
                    del tdc.child_TDCs[i]
        self.TDCs += child_TDCs

    def include_defaults(self):
        for i, tdc in enumerate(self.TDCs):
            if tdc.positivity == '':
                tdc.positivity = 'is'

                #<>TODO: figure out why this is broken
                if tdc.target[-2:] == "'s":
                    self.TDCs[i].target = self.TDCs[i].target.replace("'s",'')

    def prune_TDCs(self):
        """Prune incomplete TDCs"""
        for i, tdc in enumerate(self.TDCs):

            if tdc.type == '':
                del self.TDCs[i]
                continue

            tags = TDC.templates[tdc.type]
            parsed_spans = tdc.get_spans_from_tags(tags, include_empty_spans=False)
            empty_spans = [parsed_span[1] for parsed_span in parsed_spans
                              if len(parsed_span[0]) == 0]

            # Except the optional grounding for an action TDC
            if 'GROUNDING' in empty_spans and tdc.type == 'action':
                del self.TDCs[i].parsing['GROUNDING']
                empty_spans.remove('GROUNDING')

            # Except the optional modifier for an action TDC
            if 'MODIFIER' in empty_spans and tdc.type == 'action':
                del self.TDCs[i].parsing['MODIFIER']
                empty_spans.remove('MODIFIER')

            if len(empty_spans) > 0:
                del self.TDCs[i]


    def print_TDCs(self, print_phrases=True):
        for tdc in self.TDCs:
            if print_phrases:
                logging.info(tdc.phrase)
            logging.info(tdc)

    def plot_TDCs(self, filename='TDC Graph', scale=0.9, aspect=3.0):
        from matplotlib import rc
        rc("font", size=6)
        rc("text", usetex=False)

        num_TDCs = len(self.TDCs)
        shape = [3 * scale * aspect, scale * num_TDCs * 3.75]
        last_y = shape[1]

        # Instantiate the PGM
        pgm = daft.PGM(shape, origin=[0.2, 0.2], directed=False, aspect=aspect)

        for i, TDC in enumerate(self.TDCs):
            # Create the TDC node
            TDC_i = i + 1

            TDC_name = TDC.type + " TDC" + str(TDC_i)
            num_tags = len(TDC.parsing)
            TDC_x = scale
            TDC_y = last_y - scale * num_tags/2
            pgm.add_node(daft.Node(TDC_name, TDC_name, TDC_x, TDC_y, scale=scale))

            # Create the tags and spans
            tag_i = 0
            sorted_tags = sorted(TDC.parsing, key=lambda x:self.tag_order[x])
            for tag_, tag in enumerate(sorted_tags):
                span = TDC.parsing[tag]
                tag_name = tag + str(TDC_i)
                tag_x = TDC_x + aspect * scale * 0.7
                tag_y = last_y - scale - tag_i * 0.7 * scale
                pgm.add_node(daft.Node(tag_name, tag, tag_x, tag_y, scale=scale))

                span_name = span + str(TDC_i)
                span_x = tag_x + aspect * scale * 0.7
                span_y = tag_y
                pgm.add_node(daft.Node(span_name, span, span_x, span_y, scale=scale))

                # Add in the edges.
                pgm.add_edge(TDC_name, tag_name)
                pgm.add_edge(tag_name, span_name)
                tag_i += 1

            last_y = tag_y

        # Render and save.
        pgm.render()
        pgm.figure.savefig(filename + ".png", dpi=150)
        pgm.figure.clf()

def generate_test_data():
    data = [['Roy','TARGET' ],
        ['is','POSITIVITY'],
        ['moving','ACTION'],
        ['North','MODIFIER'],
        ['.','NULL'],
        ['That robot','TARGET'],
        ['is','POSITIVITY'],
        ['stopped','ACTION'],
        ['.','NULL'],
        ['Nothing','TARGET'],
        ['is','POSITIVITY'],
        ['next to','SPATIALRELATION'],
        ['the dresser','GROUNDING'],
        ['.','NULL'],
        ['I','NULL'],
        ['don\'t','POSITIVITY'],
        ['see','NULL'],
        ['anything','TARGET'],
        ['near','SPATIALRELATION'],
        ['the desk','GROUNDING'],
        ['.','NULL'],
        ['I think','NULL'],
        ['a robot','TARGET'],
        ['is','POSITIVITY'],
        ['in','SPATIALRELATION'],
        ['the kitchen','GROUNDING'],
        ['.','NULL'],
        ['Pris','TARGET'],
        ['is','POSITIVITY'],
        ['moving','ACTION'],
        ['really quickly','MODIFIER'],
        ['.','NULL'],
        ['The green one','TARGET'],
        ['is','POSITIVITY'],
        ['heading','ACTION'],
        ['over there','GROUNDING'],
        ['.','NULL'],
        ['The red guy','TARGET'],
        ['is','POSITIVITY'],
        ['spinning around','ACTION'],
        ['the table','GROUNDING'],
        ['.','NULL'],
        ['A robot\'s','TARGET'],
        ['moving','ACTION'],
        ['away from','MODIFIER'],
        ['you','GROUNDING'],
        ['.','NULL'],
        ['There\'s','NULL'],
        ['another robot','TARGET'],
        ['heading','ACTION'],
        ['towards','MODIFIER'],
        ['you','GROUNDING'],
        ['.','NULL'],
        ['He\'s','TARGET'],
        ['running','ACTION'],
        ['away from','MODIFIER'],
        ['you','GROUNDING'],
        ['!','NULL'],
        ['He\'s','TARGET'],
        ['behind','SPATIALRELATION'],
        ['the desk','GROUNDING'],
        [',','NULL'],
        ['about to','NULL'],
        ['leave','ACTION'],
        ['the kitchen','GROUNDING'],
        ['.','NULL'],
        ['Two robots','TARGET'],
        ['are','POSITIVITY'],
        ['moving','ACTION'],
        ['away from','MODIFIER'],
        ['each-other','GROUNDING'],
        ['.','NULL'],
        ['I','NULL'],
        ['think','NULL'],
        ['Pris','TARGET'],
        ['is','POSITIVITY'],
        ['trying to','NULL'],
        ['stay','ACTION'],
        ['in','SPATIALRELATION'],
        ['the kitchen','GROUNDING'],
        ['.','NULL'],
        ]
    return data

def generate_fleming_test_data():
    (certainties,
    positivities,
    relations,
    actions,
    modifiers,
    groundings,
    target_names) = generate_human_language_template()

    data = []
    for target in target_names:
        for positivity in positivities:

            # Spatial relation statements
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

                        # Ignore certain questons
                        if grounding_type_name == 'object':
                            if relation_name in ['inside', 'outside']:
                                continue

                        # Write spatial relation tagged data
                        data.append(['I know', 'NULL'])
                        data.append([target, 'TARGET'])
                        data.append([positivity, 'POSITIVITY'])
                        data.append([relation_name, 'SPATIALRELATION'])
                        data.append([grounding_name, 'GROUNDING'])
                        data.append(['.', 'NULL'])

                    # Action statements
                    for action in actions:
                        if action == 'stopped':
                            data.append(['I know', 'NULL'])
                            data.append([target, 'TARGET'])
                            data.append([positivity, 'POSITIVITY'])
                            data.append([action, 'ACTION'])
                            data.append(['.', 'NULL'])
                            continue

                        for modifier in modifiers:

                            data.append(['I know', 'NULL'])
                            data.append([target, 'TARGET'])
                            data.append([positivity, 'POSITIVITY'])
                            data.append([action, 'ACTION'])
                            data.append([modifier, 'MODIFIER'])
                            if modifier in ['toward', 'around']:
                                data.append([grounding_name, 'GROUNDING'])

                                str_ = ("I know " + target + ' ' + positivity + ' '
                                        + action + ' ' + modifier + ' ' +
                                        grounding_name + '.')
                            else:
                                str_ = ("I know " + target + ' ' + positivity + ' '
                                        + action + ' ' + modifier + '.')
                            data.append(['.', 'NULL'])

    return data


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    tagged_document = generate_fleming_test_data()
    TDC_collection = TDC_Collection(tagged_document)
    TDC_collection.print_TDCs()

    tagged_document = generate_fleming_test_data()

    # Plot TDCs
    # TDC_collection.plot_TDCs()