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
import numpy as np
import re

class Tokenizer(object):
    """short description of Tokenizer

    long description of Tokenizer
    
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

    def __init__(self, max_distance=2, ngram_score_discount=0):
        self.delimiters = [',', ';', '.', '!', '?']
        self.max_distance = max_distance
        self.ngram_score_discount = ngram_score_discount

        # corpus = "Roy is around the corner, inside the kitchen, near the fridge."
        # from cops_and_robots.human_tools.chatbox import DataHandler
        # corpus = "\n".join(DataHandler().input_sentences)
        # self.generate_ngram_scores(corpus)


    def tokenize(self, document, max_distance=None):
        """Takes an input document string and splits it into a list of tokens.

        Delimiter punctuation (',', ';', '.', '!', '?') is never combined with
        other tokens.

        Parameters
        ----------
        document : string
            An input string to be tokenized.
        distance : int, optional
            Upper limit of the number of words that can be combined into one
            token. Default is 1.

            For example, if max_distance=3, the sentence, "Roy is behind you."
            would have the following tokenizations:
                ['Roy', 'is', 'behind', 'you', '.']  # (i.e. max_distance=1)
                ['Roy is', 'behind', 'you', '.']  # (i.e. max_distance=2)
                ['Roy is', 'behind you', '.']  # (i.e. max_distance=2)
                ['Roy', 'is behind', 'you', '.']  # (i.e. max_distance=2)
                ['Roy', 'is', 'behind you', '.']  # (i.e. max_distance=2)
                ['Roy is behind', 'you', '.']
                ['Roy', 'is behind you', '.']

        Returns
        -------
        list
            A list of possible tokenizations, depending on the distance.
        """
        if max_distance is not None:
            self.max_distance = max_distance

        base_tokenized_document = re.findall(r"[\w']+|[.,!?;]", document)
        n_tokens = len(base_tokenized_document)

        # Find n-gram replacement tokens for all unigram tokens
        tokenized_documents = []
        tokenized_documents.append(base_tokenized_document)
        tokenized_documents += self.unigram_to_ngram(base_tokenized_document)
        
        # Ensure uniqueness
        tokenized_documents = [list(x) for x in set(tuple(x) for x in tokenized_documents)]
        tokenized_documents.sort()

        # Define scores for all n-grams > 1
        for tokenized_document in tokenized_documents:
            score = self.score_tokenized_document(tokenized_document)


        # Sort tokenized documents in score order


        return tokenized_documents

    def score_tokenized_document(self, tokenized_document, dataset=None):
        pass
        # Generate n-grams
        # http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
        # distance = 2
        # ngrams = []
        # while distance <= max_distance:
        #     grams = zip(*[dataset[i:] for i in range(distance)])
        #     ngram = [" ".join(g) for g in grams]

        #     ngrams += ngram
        #     distance += 1

    def generate_ngram_scores(self, corpus):
        """Score non-unigram word combinations based on some corpus.
        """
        # Listify dataset
        dataset = re.findall(r"[\w']+|[.,!?;]", corpus)

        # Generate all possible n-grams from the dataset
        ngram_lists = self.unigram_to_ngram(dataset)
        ngrams = dataset[:]
        for ngram_list in ngram_lists:
            for n in ngram_list:
                if n.find(" ") > -1:
                    ngrams.append(n)
        ngrams = list(set(ngrams))

        # Find number of occurrences in corpus
        self.ngram_counts = {g:0 for g in ngrams}
        for ngram, _ in self.ngram_counts.iteritems():
            self.ngram_counts[ngram] = corpus.count(ngram)

        # Score ngrams based on number of occurrences
        self.ngram_scores = {g:0 for g in ngrams}
        for ngram, count in self.ngram_counts.iteritems():

            # Find component unigrams (or ignore if already a unigram)
            unigrams = ngram.split(" ")
            if len(unigrams) < 2:
                del self.ngram_scores[ngram]
                continue

            # Calculate a score based on total number of counts
            score = count - self.ngram_score_discount
            for unigram in unigrams:
                score /= self.ngram_counts[unigram]
            self.ngram_scores[ngram] = score

    def unigram_to_ngram(self, base_tokenized_document):
        """Turn a list of unigrams into a list of ngrams.

        Replaces strings (grams) in a list of grams by their n-grams, where
        n is the max_distance. Works recursively.
        """
        tokenized_documents = []
        if self.max_distance <= 1:
            return []

        distance = 2
        while distance <= self.max_distance:
            for i in range(len(base_tokenized_document) - distance + 1):
                prev = base_tokenized_document[:i]
                gram = " ".join(base_tokenized_document[i:i + distance])
                next_ = base_tokenized_document[i + distance:]
                for delim in self.delimiters:
                    if gram.find(delim) > -1 or gram.count(" ") + 1 > self.max_distance:
                        break
                else:
                    tokenized_document = prev + [gram] + next_
                    tokenized_documents.append(tokenized_document)
                    tokenized_documents += self.unigram_to_ngram(tokenized_document)
            distance += 1
        return tokenized_documents


def test_tokenizer(document='', max_distance=3):
    if len(document) < 1:
        document = "Roy is behind you."

    tokenizer = Tokenizer(max_distance=max_distance)
    tokenized_documents = tokenizer.tokenize(document)
    for td in tokenized_documents:
        print td


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    test_tokenizer()
    # test_tokenizer("Roy is around the corner, inside the kitchen, near the fridge.", max_distance=5)