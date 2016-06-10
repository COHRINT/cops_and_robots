#!/usr/bin/env python
"""Provides a chat interface between human and robot, incorperating NLP strategies.
"""
from __future__ import division
__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import logging
import random
import numpy as np
import os

import spacy.en

from cops_and_robots.human_tools.nlp.datahandler import DataHandler
from cops_and_robots.human_tools.nlp.tagger import Tagger
from cops_and_robots.human_tools.nlp.tokenizer import Tokenizer
from cops_and_robots.human_tools.nlp.templater import TDC_Collection, TDC
from cops_and_robots.human_tools.nlp.matcher import (Matcher, template_to_string)

class Chatter(object):
    """A chat interface that maps natural language to pre-defined statements.

    Parameters
    ----------
    uncontract : bool, optional
        Whether to remove contractions in the raw input. Default is ``False``.
    punctuate : bool, optional
        Whether to manage punctuation in the raw input. Default is ``False``.
    autocorrect : bool, optional
        Whether to autocorrect words in the raw input. Default is ``False``.
    case_correct : bool, optional
        Whether to (de-)capitalize words in the raw input. Default is ``False``.

    """

    def __init__(self, uncontract=False, 
                 punctuate=False, autocorrect=False, case_correct=False):

        self.uncontract = uncontract
        self.punctuate = punctuate
        self.autocorrect = autocorrect
        self.case_correct = case_correct

        logging.info("Loading SpaCy...")
        self.nlp = spacy.en.English(parser=False)
        logging.info("Parsing corpus...")

        self.tokenizer = Tokenizer(nlp=self.nlp)
        self.tagger = Tagger()
        self.matcher = Matcher(nlp=self.nlp)

    def translate_from_natural_input(self, nl_input, n_considerations=1):
        """Transform natural language input into human sensor statement(s).

        Parameters
        ----------
        nl_input : string
            A string of natural language input. This is the raw, pre-processed
            string.
        n_considerations : int, optional
            Number of `Statement`s to propose as mappings to the *nl_input*.

        Returns
        ----------
        Statement
            A list of `Statement` objects.
        """
        statements = self._find_nearest_statement(nl_input)
        return statements

    def translate_from_console_input(self):
        """Prints responses to raw console input fed through the NLP pipeline.
        """
        while True:
            # Get natural language input
            nl_input = raw_input('Describe a target (or enter a blank line to exit): ')
            print('Your said: {}'.format(nl_input))

            if nl_input == '':
                print('Thanks for chatting!')
                break

            cleaned_document = self._clean_input_document(nl_input)
            templates = self._find_nearest_statement(cleaned_document)

            print("I understood: ")
            for template in templates:
                print("\t \"{}\".".format(template))

    def test_accuracy(self):
        dh = DataHandler()
        true_statements = dh.get_selected_statements()

        print len(dh.input_sentences), len(true_statements['jeremy']), len(true_statements['sierra'])

    def _clean_input_document(self, nl_input, uncontract=False, punctuate=False,
                             autocorrect=False, case_correct=False):
        """Performs NLP pre-processing tasks on raw natural language input.

        Parameters
        ----------
        nl_input : string
            A string of natural language input. This is the raw, pre-processed
            string.

        Returns
        ----------
        Statement
            A `Statement` object.
        """

        cleaned_document = nl_input

        #<>TODO: this!
        if self.uncontract:
            pass
        if self.punctuate:
            pass
        if self.autocorrect:
            pass
        if self.case_correct:
            pass

        return cleaned_document

    def _find_nearest_statement(self, nl_input):
        """Finds the closest human sensor statement template to the input doc.
        """
        # Clean up input text
        cleaned_document = self._clean_input_document(nl_input)

        # Identify possible word span combinations
        tokenized_document = self.tokenizer.tokenize(cleaned_document)

        # Apply CRF for semantic tagging
        tagged_document = self.tagger.tag_document(tokenized_document)

        # Separate into TDCs
        TDC_collection = TDC_Collection(tagged_document)
        filled_templates = TDC_collection.get_expected_templates()

        # Apply phrase2vec for word matching
        statements = self.matcher.find_nearest_statements(filled_templates)

        return statements



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # test_NLP_pipeline()


    chatter = Chatter()
    # chatter.test_accuracy()
    chatter.translate_from_console_input()
