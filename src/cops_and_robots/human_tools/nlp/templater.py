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
from collections import OrderedDict

from scipy.stats import dirichlet
import numpy as np

from cops_and_robots.human_tools.statement_template import StatementTemplate

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

    """

    def __init__(self, tagged_phrase='', show_children=False):
        self.tagged_phrase = tagged_phrase
        self.phrase = " ".join([s[0] for s in tagged_phrase])
        self.parsing = {}
        self.show_children = show_children
        self.child_TDCs = []

        # List possible templates
        self.templates = []
        for template_type, templates in StatementTemplate.template_trees.iteritems():
            for j, template_keys in enumerate(templates):
                empty_template = OrderedDict(zip(template_keys,
                                                 [''] * len(template_keys)))
                t = (template_type + str(j), empty_template)
                self.templates.append(t)
        self.template_counts = [1] * len(self.templates)

        # Generate likelihoods of each template based on the input phrase
        self.fit_phrase()
        self.fill_in_defaults()
        self.prune()

    def __repr__(self):
        expected_template = self.get_expected_templates()[0]
        str_ = "{} TDC: {}".format(expected_template[0].title(), expected_template[1])
        # if self.show_children:
        #     if len(self.child_TDCs) > 0:
        #         str_ += " with children: "
        #         for child_TDC in self.child_TDCs:
        #             str_ += child_TDC.__repr__()
        return str_

    def fit_phrase(self):
        """Fit the tagged_phrase into possible templates

        This also generates counts of observed labels for each template,
        where the counts are normalized by the template lengths. Each template
        is modeled as its own dirichlet distribution with dimension equal to
        the number of labels in the template. The expected values of these
        dirichlets can be compared to identify the most likely template(s).
        """
        for i, template in enumerate(self.templates):
            for tagged_token in self.tagged_phrase:
                token = tagged_token[0]
                tag = tagged_token[1]

                for template_key in template[1].keys():

                    #<>TODO: Fix CRF labels instead of this modification
                    key = template_key
                    if 'spatial_relation' in template_key:
                        key = template_key.replace('_', '')

                    # Fill template and add count (normalized by template length)
                    if tag.upper() in key.upper():
                        if template[1][template_key] != '':
                            self.add_to_child_TDC(tag, token)
                        else:
                            template[1][template_key] = token
                            self.template_counts[i] += 1 / len(template[1])
                            #<>TODO: validate normalization over temp. length

        self.template_expected_probs = dirichlet.mean(self.template_counts)

    def add_to_child_TDC(self, tag, token):
        # If no child TDCs exist, make new child TDC

        # If child TDCs exist, add tag/token pair to first empty one

        # Add in default values

        # Take parent values
        pass

    def fill_in_defaults(self):
        for template in self.templates:
            for template_key in template[1].keys():
                if template[1][template_key] == '':
                    try:
                        def_ = StatementTemplate.default_components[template_key]
                        template[1][template_key] = def_
                    except KeyError:
                        logging.debug("no {} default".format(template_key))

    def prune(self):
        for template in self.templates:
            for template_key in template[1].keys():
                    if template_key == '':
                        del template[1]['']

    def get_expected_templates(self):
        # Find the most likely template(s) as the suggested TDC
        i = np.argmax(self.template_expected_probs)
        templates = [self.templates[i]]
        for child in self.child_TDCs:
            templates.append(child.get_expected_templates())
        return templates


    # def _check_input(self, input_span, true_span_tag, TDC_type=''):
    #     # Check for template matching
    #     is_conflicting = ((true_span_tag not in self.template) 
    #                       and len(self.template) > 0)
        
    #     # Check for duplicate spans
    #     current_span = self.get_spans_from_tags([true_span_tag])[0][0]
    #     has_duplicate = len(current_span) > 0
        
    #     # Add to/new child TDC if necessary
    #     if is_conflicting or has_duplicate:
    #         if len(self.child_TDCs) > 0:
    #             self.child_TDCs[0].add_span([input_span, true_span_tag])
    #         else:    
    #             new_tdc = TDC(type_=TDC_type, phrase=self.phrase)
    #             new_tdc.add_span([input_span, true_span_tag])
    #             logging.debug("Adding a child TDC to this {} --> {}"
    #                           .format(self, new_tdc))
    #             self.child_TDCs.append(new_tdc)
    #         return False
        
    #     # Set template and type
    #     if len(TDC_type) > 0:
    #         self.type = TDC_type
    #     if len(self.type) > 0:
    #         self.template = TDC.templates[self.type]
            
    #     return True
        
    # def add_span(self, tagged_span=[], span='', span_tag=''):
    #     if len(tagged_span) > 0:
    #         span, span_tag = tagged_span
            
    #     if span_tag == 'NULL':
    #         logging.debug("Skipping null spans.")
    #         return
            
    #     if len(tagged_span) == 1 or len(tagged_span) > 2:
    #         logging.warning("You must specify a word and its semantic tag!")
    #         return
            
    #     if span_tag == 'TARGET':
    #         self.target = span
    #     elif span_tag == 'POSITIVITY':
    #         self.positivity = span
    #     elif span_tag == 'GROUNDING':
    #         self.grounding = span
    #     elif span_tag == 'SPATIALRELATION':
    #         self.spatial_relation = span
    #     elif span_tag == 'ACTION':
    #         self.action = span
    #     elif span_tag == 'MODIFIER':
    #         self.modifier = span
        
    # def get_spans_from_tags(self, span_tag, include_empty_spans=True):
    #     if type(span_tag) is not type(list()):
    #         span_tag = [span_tag]
    #     parsed_spans = []
        
    #     for span_tag in span_tag:
    #         if span_tag == 'TARGET':
    #             span = self.target
    #         elif span_tag == 'POSITIVITY':
    #             span = self.positivity
    #         elif span_tag == 'GROUNDING':
    #             span = self.grounding
    #         elif span_tag == 'SPATIALRELATION':
    #             span = self.spatial_relation
    #         elif span_tag == 'ACTION':
    #             span = self.action
    #         elif span_tag == 'MODIFIER':
    #             span = self.modifier
                
    #         if not include_empty_spans and span == '':
    #             continue

    #         parsed_spans.append([span, span_tag])
    #     return parsed_spans

    # def to_tagged_phrase(self):
    #     return [[word, tag] for tag, word in self.parsing.iteritems()]


class TDC_Collection(object):
    """A collection of TDCs

    """

    def __init__(self, tagged_document):
        self.TDCs = []
        self.parse_tagged_document(tagged_document)

        self.tag_order = {'TARGET': 0,
                          'POSITIVITY': 1,
                          'SPATIALRELATION': 2,
                          'ACTION': 3,
                          'MODIFIER': 4,
                          'GROUNDING': 5,
                          }

    def parse_tagged_document(self, tagged_document):
        """Generate a complete parsing of a tagged document.

        Parameters
        ----------
        tagged_document : array_like
            A 2-by-n array of tokens and tags.
        """
        tagged_phrases = self.split_phrases(tagged_document)
        self.get_TDCs_from_phrases(tagged_phrases)
        self.flatten_TDCs() #<>TODO
        self.prune_TDCs() #<>TODO

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

        if len(phrase) > 0:
            phrases.append(phrase)

        return phrases

    def get_TDCs_from_phrases(self, tagged_phrases):
        """Create TDCs for each phrase"""
        for tagged_phrase in tagged_phrases:
            tdc = TDC(tagged_phrase=tagged_phrase)
            self.TDCs.append(tdc)

    def flatten_TDCs(self, remove_children=False):
        """Flatten TDC list, optionally removing children"""
        return
        child_TDCs = []
        for tdc in self.TDCs:
            for i, child_TDC in enumerate(tdc.child_TDCs):
                child_TDCs.append(child_TDC)
                if remove_children:
                    del tdc.child_TDCs[i]
        self.TDCs += child_TDCs

    def prune_TDCs(self):
        """Prune incomplete TDCs"""
        return
        for i, tdc in enumerate(self.TDCs):

            if tdc.type == '':
                del self.TDCs[i]
                continue

            tags = TDC.templates[tdc.type]
            parsed_spans = tdc.get_spans_from_tags(tags, include_empty_spans=False)
            empty_spans = [parsed_span[1] for parsed_span in parsed_spans
                              if len(parsed_span[0]) == 0]


            if len(empty_spans) > 0:
                del self.TDCs[i]

    def print_TDCs(self, print_phrases=True):
        for tdc in self.TDCs:
            if print_phrases:
                logging.info(tdc.phrase)
            logging.info(tdc)
            logging.info('')

    def plot_TDCs(self, filename='TDC Graph', scale=0.9, aspect=3.0):
        from matplotlib import rc
        import daft

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

    def get_expected_templates(self):
        expected_templates = []
        for tdc in self.TDCs:
            expected_templates.append(tdc.get_expected_templates()[0])
        return expected_templates

def test_TDC():
    from cops_and_robots.human_tools.nlp.tagger import generate_test_data
    tagged_document = generate_test_data()
    phrase = tagged_document[9:14]  # Nothing is next to the dresser
    # phrase = tagged_document[42:47] # A robot's heading away from you
    tdc = TDC(phrase)
    print phrase
    print "expected template", tdc.get_expected_templates()

def test_TDC_collection():
    from cops_and_robots.human_tools.nlp.tagger import generate_test_data
    tagged_document = generate_test_data()
    TDC_collection = TDC_Collection(tagged_document)
    # TDC_collection.print_TDCs()
    filled_templates = TDC_collection.get_expected_templates()
    return filled_templates

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    

    # test_TDC()
    print test_TDC_collection()