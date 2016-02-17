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

from cops_and_robots.fusion.word_matching import SimilarityChecker, template_to_string

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

            if data = '':
                print('Thanks for chatting!')
                break

            # Identify possible word span combinations
            #<>TODO: do this!

            # Apply CRF for semantic tagging
            tagged_document = self.tagger.tag_document(document)

            # Separate into TDCs
            self.TDC_collection = TDC_Collection(tagged_document)
            TDC_collection.print_TDCs()

            # Apply word2vec for word matching
            template_phrases = []
            print("I heard: ")
            for i, TDC in enumerate(TDC_collection.TDCs):
                tagged_phrase = TDC.to_tagged_phrase()
                t = self.similarity_checker.find_closest_phrase(tagged_phrase, TDC.type)
                template_phrases.append(t)

                print(template_to_string(t))
                if 1 < i < len(TDC_collection.TDCs):
                    print("\n and \n")


class ConditionalRandomField(object):
    """short description of ConditionalRandomField

    long description of ConditionalRandomField
    
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

    def __init__(self,param):

        self._generate_crf_model()


def _generate_crf_model(self):
    if 
    self._create_template()

    # Training and test data
    training_data = generate_fleming_test_data()
    test_data = generate_test_data()

    # Save to a file
    # training_data = data[:len(data)//2]
    # test_data = data[len(data)//2:]
    with open('training.data','w') as file_:
        for d in training_data:
            str_ = '\t'.join(d) + '\n'
            str_ = str_.replace (" ", "_")
            file_.write(str_)
    with open('test.data','w') as file_:
        for d in test_data:
            str_ = '\t'.join(d) + '\n'
            str_ = str_.replace (" ", "_")
            file_.write(str_)
        

        
# Create Template
def _create_template(model_i=1):
    model0 = """
               # Unigram
               U00:%x[-2,0]
               U01:%x[-1,0]
               U02:%x[0,0]
               U03:%x[1,0]
               U04:%x[2,0]
               U05:%x[-1,0]/%x[0,0]
               U06:%x[0,0]/%x[1,0]
               
               # Bigram
               B"""

    model1 = """
               # Unigram
               U00:%x[-2,0]
               U01:%x[-1,0]
               U02:%x[0,0]
               U03:%x[1,0]
               U04:%x[2,0]
               U05:%x[-1,0]/%x[0,0]
               U06:%x[0,0]/%x[1,0]
                              
               # Bigram
               B"""
    models = [model0, model1]
    template = models[model_i]
  
    with open('template','w') as file_:
        file_.write(textwrap.dedent(template))

    def tag_natural_language(self, document):
    # 

    # 
    
    return tagged_document

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    chatbox = Chatbox()
    chatbox.translate_from_natural_input()

