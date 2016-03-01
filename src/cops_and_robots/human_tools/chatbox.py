#!/usr/bin/env python
"""Provides a chat interface between human and robot, incorperating NLP strategies.
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

import textwrap

from cops_and_robots.human_tools.word_matching import SimilarityChecker, template_to_string
from cops_and_robots.human_tools.conditional_random_fields import ConditionalRandomField
from cops_and_robots.human_tools.template_description_clauses import TDC_Collection, TDC

class Chatbox(object):
    """short description of Chatbox

    long description of Chatbox
    
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
        self.similarity_checker = SimilarityChecker()
        self.tagger = ConditionalRandomField()


    def translate_from_natural_input(self):
        while True:
            # Get natural language input
            nl_input = raw_input('Describe a target (or enter a blank line to exit): ')
            print('Your said: {}'.format(nl_input))

            if nl_input == '':
                print('Thanks for chatting!')
                break

            # Identify possible word span combinations
            #<>TODO: do this!
            document = nl_input

            # Apply CRF for semantic tagging
            tagged_document = self.tagger.tag_document(document)

            # Separate into TDCs
            self.TDC_collection = TDC_Collection(tagged_document)

            # Apply word2vec for word matching
            template_phrases = []
            for i, TDC in enumerate(self.TDC_collection.TDCs):
                tagged_phrase = TDC.to_tagged_phrase()
                print tagged_phrase
                t = self.similarity_checker.find_closest_phrase(tagged_phrase, TDC.type)
                template_phrases.append(t)

                print(template_to_string(t))
                if 1 < i < len(self.TDC_collection.TDCs):
                    print("\n and \n")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    chatbox = Chatbox()
    chatbox.translate_from_natural_input()

