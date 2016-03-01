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
from itertools import chain
from collections import OrderedDict

import numpy as np
import spacy.en

from cops_and_robots.human_tools.human import generate_human_language_template

class SimilarityChecker(object):
    """short description of SimilarityChecker

    long description of SimilarityChecker
    
    Parameters
    ----------
    param : param_type, optional
        param_description

    Attributes
    ----------
    attr : attr_type
        attr_description

    Methods
    ----------
    attr : attr_type
        attr_description

    """

    def __init__(self):
        logging.info("Loading SpaCy...")
        self.nlp = spacy.en.English(parser=False, tagger=False)
        logging.info("Done!")

        self.templates = generate_human_language_template()._asdict()
        self._flatten_templates()
        self._create_phrase_templates()

    def _flatten_templates(self):
        flat_templates = {}
        for key, value in self.templates.iteritems():
            if isinstance(value, dict):
                value = list(chain(*value.values()))

            #<>TODO: match keys to *unified* variable names
            if key == 'target_names':
                key = 'target'
            if key == 'positivities':
                key = 'positivity'
            if key == 'certainties':
                key = 'certainty'
            if key == 'relations':
                key = 'spatialrelation'
            if key == 'actions':
                key = 'action'
            if key == 'modifiers':
                key = 'modifier'
            if key == 'groundings':
                key = 'grounding'

            flat_templates[key.upper()] = value
        self.templates = flat_templates

    def _create_phrase_templates(self):
        self.phrase_templates = {}
        d = OrderedDict([('I', 'I'),
                         ('CERTAINTY', 'know'),
                         ('TARGET', ''),
                         ('POSITIVITY', ''),
                         ('GROUNDING', ''),
                         ('SPATIALRELATION', ''),
                         ('.', '.'),
                         ])
        self.phrase_templates['spatial relation'] = d
        d = OrderedDict([('I', 'I'),
                         ('CERTAINTY', 'know'),
                         ('TARGET', ''),
                         ('POSITIVITY', ''),
                         ('ACTION', ''),
                         ('MODIFIER', ''),
                         ('GROUNDING', ''),
                         ('.', '.'),
                         ])
        self.phrase_templates['action'] = d

    def find_closest_phrase(self, tagged_phrase, template_type):
        """Find closest matching template phrases to input phrase.
        """
        # For each tagged word (word span) in the tagged phrase
        #<>TODO: break the independence assumption! We're assuming SRs and
        # Groundings are independent of one-another, for instance. Not true!            
        template = self.phrase_templates[template_type].copy()
        for tagged_word_span in tagged_phrase:
            tag = tagged_word_span[1]
            word_span = tagged_word_span[0]
            if tag == 'NULL':
                continue

            closest_word = self.find_closest_word(tag, word_span)
            template[tag] = closest_word

        # Delet empty values
        for key, value in template.iteritems():
            if value == '':
                del template[key]

        return template

    def find_closest_word(self, tag, word_span):
        """Find the closest template word to an individual word-span, given a tag.
        """
        similarities = []

        for template_word in self.templates[tag]:
            tw = self.nlp(unicode(template_word))
            ws = self.nlp(unicode(word_span.replace('_',' ')))
            s = tw.similarity(ws)
            similarities.append(s)

        ranked_template_words = [x for (y,x) in 
            sorted(zip(similarities, self.templates[tag]), reverse=True)]

        logging.debug("\nSimilarity to '{}'".format(word_span))
        for word, similarity in zip(self.templates[tag], similarities):
            logging.debug("\t\t'{}': {}".format(word, similarity))
        return ranked_template_words[0]

def template_to_string(template):
    str_ = " ".join(filter(None, template.values()[:-1]))
    str_ += template.values()[-1]
    return str_

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    sc = SimilarityChecker()

    tagged_phrase = [['Roy','TARGET'],
                     ['is','POSITIVITY'],
                     ['moving','ACTION'],
                     ['North','MODIFIER'],
                     ['.','NULL']]
    template_type = 'action'

    template = sc.find_closest_phrase(tagged_phrase, template_type)