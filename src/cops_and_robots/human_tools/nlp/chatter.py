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
import pandas as pd
import re
from StringIO import StringIO  # got moved to io in python3.
import requests
import os


from cops_and_robots.human_tools.nlp.tagger import Tagger
from cops_and_robots.human_tools.nlp.tokenizer import Tokenizer
from cops_and_robots.human_tools.nlp.templater import TDC_Collection, TDC
from cops_and_robots.human_tools.nlp.similarity_checker import (SimilarityChecker,
                                                            template_to_string)

class Chatter(object):
    """A chat interface that maps natural language to template sensor statements.

    """

    def __init__(self, skip_similarity=False):
        self.tokenizer = Tokenizer()
        self.tagger = Tagger()
        if not skip_similarity:
            self.similarity_checker = SimilarityChecker()

    def translate_from_natural_input(self, nl_input=''):
        if nl_input == '':
            self.translate_from_console_input()
        else:
            phrase, template = self.find_closest_template_phrase(nl_input)
            return phrase, template

    def translate_from_console_input(self):
        while True:
            # Get natural language input
            nl_input = raw_input('Describe a target (or enter a blank line to exit): ')
            print('Your said: {}'.format(nl_input))

            if nl_input == '':
                print('Thanks for chatting!')
                break

            templates = self.find_closest_template_phrase(cleaned_document)

            print("I understood: ")
            for template in templates:
                print("\t \"{}\", with probability {}."
                      .format(template[0],template[1]))

    def find_closest_template_phrase(self, nl_input):
        """Finds the closest human sensor statement template to the input doc.
        """
        # Clean up input text
        cleaned_document = self.clean_document(nl_input)

        # Identify possible word span combinations
        tokenized_documents = self.tokenizer.tokenize(cleaned_document, max_distance=1)
        
        #<>STUB: include only first tokenization of the document
        tokenized_document = tokenized_documents[0]

        # Apply CRF for semantic tagging
        tagged_document = self.tagger.tag_document(tokenized_document)

        # Separate into TDCs
        TDC_collection = TDC_Collection(tagged_document)

        # Apply word2vec for word matching
        closest = self.similarity_checker.find_closest_phrases(TDC_collection)

        # <>STUB handling only the one closest phrase
        phrase, template = closest[0]

        return phrase, template

    def clean_document(self, nl_input):
        #<>TODO: this!
        cleaned_document = nl_input
        return cleaned_document


class DataHandler(object):
    """docstring for DataHandler"""

    def __init__(self, data_url=''):
        # Get data from google sheet associated with a specific URL
        if len(data_url) < 1:
            self.url = 'https://docs.google.com/spreadsheet/ccc?key=1S-nYlSuQGCbfUTTw2DKebJ6xZJRmya3RklXC9MpjbBk&gid=2089135022&output=csv'
        else:
            self.url = data_url

        data = requests.get(self.url).content

        # Find the input sentences
        self.df = pd.read_csv(StringIO(data))
        self.input_sentences = [a for a in self.df['Input Sentence']
                                if isinstance(a, str)]

        # Find the proper single word tokenization
        single_word_tokens = [re.findall(r"[\w']+|[.,!?;]", a)
                              for a in self.input_sentences]
        single_word_tokens = [b for a in single_word_tokens for b in a]

        # Tokenize punctuation as well
        punct = [".",",","!","?",";"]
        n = len(single_word_tokens); i = 0
        while i < n - 1:
            token = single_word_tokens[i]
            next_token = single_word_tokens[i + 1]
            
            if token in punct and next_token in punct:
                del single_word_tokens[i + 1]
                n -= 1
            else:
                i += 1

        # Merge SWT with the main dataframe
        try:
            del self.df['Approx. Single Word Tokens']
        except KeyError:
            logging.debug("No column to delete")
        self.df["Single Word Tokens"] = pd.Series(single_word_tokens,
                                                  index=self.df.index)
        cols = self.df.columns.tolist()
        cols = [cols[0]] + [cols[-1]] + cols[1:-1]
        self.df = self.df[cols]


def test_NLP_pipeline():

    # Grab data, remember indices of training/test data #######################
    dh = DataHandler()

    # Find ids associated with each sentence group
    j = 0
    sentence_ids = []
    for i, row in dh.df.iterrows():
        if isinstance(row['Input Sentence'], str):
            j += 1
        sentence_ids.append(j)

    # Split corpus into training and test sets
    corpus = dh.input_sentences
    corpus = [[i,s] for i,s in enumerate(dh.input_sentences)]
    random.shuffle(corpus)
    n = len(corpus) // 2
    training_corpus = corpus[:n]
    test_corpus = corpus[n:]
    training_ids = [d[0] for d in training_corpus]

    # Tokenize test corpus (unigram model)
    input_document = "\n".join([d[1] for d in test_corpus])
    tokenizer = Tokenizer(max_distance=1)
    tokenized_test_corpus = tokenizer.tokenize((input_document))[0]

    # Get Sierra's and Jeremy's trained taggers ###############################

    # Grab the full training data
    sierra_full_data = zip(dh.df["Sierra's Tokens"].tolist(),
                           dh.df["Sierra's Labels"].tolist())
    jeremy_full_data = zip(dh.df["Jeremy's Tokens"].tolist(),
                           dh.df["Jeremy's Labels"].tolist())

    # Limit J&S's training data to the randomized corpus
    sierra_td = [d for i, d in enumerate(sierra_full_data)
                 if sentence_ids[i] in training_ids]
    jeremy_td = [d for i, d in enumerate(jeremy_full_data)
                 if sentence_ids[i] in training_ids]

    # Ignore NaN lines
    sierra_td = [d for d in sierra_td if not (isinstance(d[0], float)
                or isinstance(d[1], float))]
    jeremy_td = [d for d in jeremy_td if not (isinstance(d[0], float)
                or isinstance(d[1], float))]

    # Prepare training files and paths
    data_dir = os.path.dirname(__file__) + '/data/'
    sierra_training_file = "sierra_training.txt"
    jeremy_training_file = "jeremy_training.txt"
    sierra_training_path = data_dir + sierra_training_file
    jeremy_training_path = data_dir + jeremy_training_file

    # Write training files
    for tf, td in zip([sierra_training_path, jeremy_training_path],
                      [sierra_td, jeremy_td]):
        with open(tf,'w') as file_:
            for d in td:
                str_ = '\t'.join(d) + '\n'
                str_ = str_.replace (" ", "_")
                file_.write(str_)

    # Train tagging engine using both Sierra's and Jeremy's tags
    s_tagger = Tagger(training_file='sierra_training.txt',
                      test_file='sierra_test.txt',
                      input_file='sierra_input.txt',
                      model_file='sierra_model.txt',
                      output_file='sierra_output.txt',
                      )
    j_tagger = Tagger(training_file='jeremy_training.txt',
                      test_file='jeremy_test.txt',
                      input_file='jeremy_input.txt',
                      model_file='jeremy_model.txt',
                      output_file='jeremy_output.txt',
                      )

    # Evaluate tagging agreement ##############################################
    s_tagged_document = s_tagger.tag_document(tokenized_test_corpus)
    j_tagged_document = j_tagger.tag_document(tokenized_test_corpus)

    agreements = []
    for i, _ in enumerate(s_tagged_document):
        agreement = s_tagged_document[i][1] == j_tagged_document[i][1]
        if agreement:
            agreements.append(1)
        else:
            agreements.append(0)
    agreements = np.array(agreements)

    print agreements.mean()

    # Generate TDCs ###########################################################
    # s_TDCs = TDC_Collection(s_tagged_document)
    # j_TDCs = TDC_Collection(j_tagged_document)
    # j_TDCs.plot_TDCs()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # test_NLP_pipeline()


    chatter = Chatter()
    chatter.translate_from_natural_input()