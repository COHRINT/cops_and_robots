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
import re
import os
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
import spacy.en
from scipy.misc import logsumexp

from cops_and_robots.human_tools.nlp.datahandler import DataHandler
from cops_and_robots.map_tools.layer import fake_parula_cmap

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

    def __init__(self, max_distance=2, ngram_score_discount=0, nlp=None):
        self.delimiters = [',', ';', '.', '!', '?']
        self.breaking_delimeters = ['.', '!', '?']
        self.max_distance = max_distance
        self.scorer = Scorer(load_data=True, nlp=nlp)

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

        #<>TODO: make customizable string of regex format
        singly_tokenized_document = re.findall(r"[\w']+|[.,!?;]", document)
        n_tokens = len(singly_tokenized_document)

        tokenized_document = []
        str_ = singly_tokenized_document[0]
        for i, u in enumerate(singly_tokenized_document[:-1]):
            v = singly_tokenized_document[i + 1]

            is_grouped = self.scorer.classify_word_pair((u,v))

            if u in self.breaking_delimeters or v in self.breaking_delimeters:
                is_grouped = False

            if is_grouped:
                str_ += " " + v
            else:
                tokenized_document.append(str_)
                str_ = v
        tokenized_document.append(str_)

        return tokenized_document

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

    def test_accuracy(self):
        dh = DataHandler()
        multi_word_tokens = dh.get_multi_word_tokens()

        # Generate tokenization of entire document
        document = dh.get_input_sentences()
        tokenized_document = self.tokenize(document)

        st = multi_word_tokens['sierra']
        jt = multi_word_tokens['jeremy']
        ht = tokenized_document

        j_end, s_end = 0,0
        validation = np.zeros(len(ht))
        for i, token in enumerate(ht):
            # Cycle individual sentences (once token is a breaking delim)
            if i == 0 or token in self.breaking_delimeters:
                if j_end < len(jt):
                    j_start = j_end
                j = j_start
                while jt[j] not in self.breaking_delimeters and j < len(jt):
                    j += 1
                j_end = j + 1

                if s_end < len(st):
                    s_start = s_end
                s = s_start
                while st[s] not in self.breaking_delimeters and s < len(st):
                    s += 1
                s_end = s + 1

            # Compare with all jeremy's tokens in range
            for j in range(j_start, j_end):
                if jt[j] == token:
                    validation[i] = 1

            # Compare with all sierra's tokens in range
            for s in range(s_start, s_end):
                if st[s] == token:
                    validation[i] = 1
        accuracy = validation.sum() / validation.size
        logging.info("Tokenizer accuracy: {}%".format(accuracy * 100))

class Scorer(object):
    """Scores a corpus of data based on lemmata and/or parts-of-speech tags.

    Parts of speech: https://sites.google.com/site/partofspeechhelp/home
    """

    def __init__(self, corpus=None, load_data=True, save_data=True, nlp=None):

        self.data_dir = os.path.dirname(os.path.abspath(__file__)) + '/data/'
        self.filename = self.data_dir + 'score_data.npz'

        if nlp is None:
            logging.info("Loading SpaCy...")
            self.nlp = spacy.en.English(parser=False)
            logging.info("Parsing corpus...")
        else:
            self.nlp = nlp

        self.feature_type_values = OrderedDict({'lemmata': None,
                                                'POS tags': None,
                                                })

        # Training parameters (overwritten on load)
        self.num_classes = 2
        self.weight_decay = 1e-3
        self.max_iter = 1e6
        self.step_size = 1e-2
        self.min_step = 1e-5
        self.whiten = False

        if load_data:
            self._load_data()
        else:
            self._parse_corpus(corpus)
            self.learning_params = {'num classes': self.num_classes,
                                    'weight decay': self.weight_decay,
                                    'max iter': self.max_iter,
                                    'step size': self.step_size,
                                    'min step': self.min_step,
                                    'whiten': self.whiten,
                                    }

            logging.info("Creating contingency tables...")
            self._create_contingency_tables()

            logging.info("Generating features and labels from truth data...")
            self._parse_truth_data()
            self._generate_features_and_labels()

            logging.info("Learning weights...")
            self._learn_weights()
            if save_data:
                self._save_data()

    def view_truth_data(self):
        from matplotlib.widgets import RadioButtons

        data = self.feature_vectors[:,1:]

        labels = []
        for score_type in Scorer.score_functions.keys():
            for feature_type in ['lemmata', 'POS']:
                labels.append(score_type + ' (' + feature_type + ')')


        select = [0,0]

        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        scat_grouped = ax.scatter(0, 0, color='r')
        scat_ungrouped = ax.scatter(0, 0, color='b')
        plt.subplots_adjust(left=0.4)

        # Selectors
        def select_data(select):
            x_data = data[:, select[0]]
            x_min, x_max = np.nanmin(x_data), np.nanmax(x_data)
            x_label = labels[select[0]]

            y_data = data[:, select[1]]
            y_min, y_max = np.nanmin(y_data), np.nanmax(y_data)
            y_label = Scorer.score_functions.keys()[np.mod(select[1] - 1, 2)]
            y_label = labels[select[1]]

            # truth_labels = np.random.randint(0, 2, datasize)
            truth_labels = self.class_ids

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))
            scat_grouped.set_offsets(zip(x_data[truth_labels==1], y_data[truth_labels==1]))
            scat_ungrouped.set_offsets(zip(x_data[truth_labels==0], y_data[truth_labels==0]))
            plt.draw()

        def select_x(label):
            select[0] = labels.index(label)
            select_data(select)

        def select_y(label):
            select[1] = labels.index(label)
            select_data(select)
            
        rax = plt.axes([0.025, 0.025, 0.275, 0.45], frameon=True,)# aspect='equal')
        radio_x = RadioButtons(rax, labels)
        for circle in radio_x.circles:
            circle.set_radius(0.02)
        radio_x.on_clicked(select_x)

        rax = plt.axes([0.025, 0.5, 0.275, 0.45], frameon=True,)# aspect='equal')
        radio_y = RadioButtons(rax, labels)
        for circle in radio_y.circles:
            circle.set_radius(0.02)
        radio_y.on_clicked(select_y)


        logging.info(self.learning_params)
        plt.show()

    def visualize_score(self, score_matrix, feature_type, name='', divergent=False):
        fig = plt.figure(figsize=(14,8))
        ax = fig.add_subplot(111)

        if divergent:
            cmap = plt.get_cmap('RdYlGn')
        else:
            cmap = fake_parula_cmap()

        cax = ax.imshow(score_matrix, interpolation='nearest', picker=True,
                        cmap=cmap)
        fig.colorbar(cax)

        ax.set_xlabel('Second word in pair')
        ax.set_ylabel('First word in pair')

        def onpick(event):
            xi, yi = (int(round(n)) for n in (event.mouseevent.xdata,
                                              event.mouseevent.ydata))
            str_ = ("U='{}', V='{}', value={:.3f}"
                    .format(self.feature_type_values[feature_type][yi],
                            self.feature_type_values[feature_type][xi],
                            score_matrix[yi,xi]
                            )
                    )

            try:
                self.text.remove()
            except:
                pass

            bbox = {'facecolor': 'white',
                    'alpha': 0.8,
                    'boxstyle':'round',
                    }
            self.text = ax.annotate(str_, xy=(0.3, -0.1), bbox=bbox,
                                    xycoords='axes fraction', fontsize=16,
                                    annotation_clip=False,
                                    )
            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', onpick)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if len(name) > 0:
            ax.set_title(name)

    def test_accuracy(self):

        # Grab test feature vectors
        n = len(self.test_data_indices)
        feature_vectors = self.feature_vectors[self.test_data_indices]
        labels = self.labels[self.test_data_indices]
        feature_vectors[np.isinf(feature_vectors)] = 0

        # Generate hypotheses
        M = self.weights.T .dot (feature_vectors.T)
        b = np.nanmax(M)  # To deal with overflow
        log_hyp = M - logsumexp(M, axis=0)
        hypothesis = np.exp(log_hyp)
        hypothesis[np.isnan(hypothesis)] = 0.5  # <>TODO: remove cheap hack!

        # Check accuracy
        predictions = np.argmax(hypothesis, axis=0)
        truth = self.class_ids[self.test_data_indices]
        validation = np.equal(predictions, truth).astype(int)
        accuracy = validation.sum() / n

        # Check precision
        correct_predicted_groups = np.array([i for i, v in enumerate(predictions)
                                             if v and v==truth[i]])
        all_predicted_groups = predictions[predictions==1]
        precision = correct_predicted_groups.size / all_predicted_groups.size

        # Check recall - the number of true grouped word over all grouped words
        all_true_groups = truth[truth==1]
        recall = correct_predicted_groups.size / all_true_groups.size

        logging.info('{:0.2f}% accuracy'.format(accuracy * 100))
        logging.info('{:0.2f}% precision'.format(precision * 100))
        logging.info('{:0.2f}% recall'.format(recall * 100))

        return accuracy

    def classify_word_pair(self, word_pair):

        score = self.score_words(word_pair)
        score_vector = self.score_vector_from_scores(score)
        feature_vector = np.hstack((1, score_vector))


        # Generate Hypothesis
        M = self.weights.T .dot (feature_vector.T)
        log_hyp = M - logsumexp(M, axis=0)
        hypothesis = np.exp(log_hyp)
        hypothesis[np.isnan(hypothesis)] = 0.5  # <>TODO: remove cheap hack!

        prediction = np.argmax(hypothesis)
        return bool(prediction)

    def score_words(self, word_pair=None, visualize=False, feature_type=None):
        """

        If word_pair is none, score the entire corpus.
        """
        if feature_type is None:
            types = self.feature_type_values.keys()
        else:
            types = [feature_type]

        scores = {feature_type: {} for feature_type in types}
        for feature_type in types:
            for score_type, score_function in Scorer.score_functions.iteritems():
                score = score_function(self,
                                       word_pair=word_pair,
                                       feature_type=feature_type,
                                       visualize=visualize,
                                       )
                scores[feature_type][score_type] = score

        return scores

    def score_vector_from_scores(self, scores):
        score_vec = []
        i = 0
        for feature_type, _ in scores.iteritems():
            for score_type, _ in scores[feature_type].iteritems():
                try:
                    s = scores[feature_type][score_type]
                except KeyError:
                    s = np.nan
                score_vec.append(s)
                i += 1

        return np.array(score_vec)

    def get_word_pair_measure(self, word_pair, feature_type, freq_measures):
        """Returns a frequency measure contingincy table for a word pair.
        """
        word_pair_features = []
        doc = self.nlp(unicode(" ".join(word_pair)))

        if feature_type == 'lemmata':
            word_pair_features = [t.lemma_ for t in doc]
        elif feature_type == 'POS tags':
            word_pair_features = [t.tag_ for t in doc]

        feature_list = self.feature_type_values[feature_type]
        # print feature_list
        try:
            u = feature_list.index(word_pair_features[0])
        except ValueError:
            u = None

        try:
            v = feature_list.index(word_pair_features[1])
        except ValueError:
            v = None

        if u is None or v is None:
            return [None] * len(freq_measures)
        else:
            ret = []
            for freq_measure in freq_measures:
                ret.append(freq_measure[feature_type][u,v,:])
            if len(ret) == 1:
                ret = ret[0]
            return ret

    def t_score(self, feature_type, word_pair=None, visualize=False):
        if word_pair is None:
            O_11 = self.observed_freqs[feature_type][:,:,0]
            E_11 = self.expected_freqs[feature_type][:,:,0]
        else:
            freq_measures = [self.observed_freqs, self.expected_freqs]
            O, E = self.get_word_pair_measure(word_pair, feature_type, freq_measures)
            if O is None or E is None:
                O_11, E_11 = np.nan, np.nan
            else:
                O_11, E_11 = O[0], E[0]
        score = (O_11 - E_11) / np.sqrt(O_11)
        # score[np.isnan(score)] = 0

        if visualize:
            self.visualize_score(score, feature_type, "T-score", divergent=True)
        return score

    def z_score(self, feature_type, word_pair=None, visualize=False):
        if word_pair is None:
            O_11 = self.observed_freqs[feature_type][:,:,0]
            E_11 = self.expected_freqs[feature_type][:,:,0]
        else:
            freq_measures = [self.observed_freqs, self.expected_freqs]
            O, E = self.get_word_pair_measure(word_pair, feature_type, freq_measures)
            if O is None or E is None:
                O_11, E_11 = np.nan, np.nan
            else:
                O_11, E_11 = O[0], E[0]
        score = (O_11 - E_11) / np.sqrt(E_11)
        # score[np.isnan(score)] = 0

        if visualize:
            self.visualize_score(score, feature_type, "Z-score", divergent=True)
        return score

    def chi_square_score(self, feature_type, word_pair=None, visualize=False):
        if word_pair is None:
            score = np.zeros_like(self.observed_freqs[feature_type][:,:,0])
        else:
            score = 0

        for i in np.arange(4):
            if word_pair is None:
                O_ij = self.observed_freqs[feature_type][:,:,i]
                E_ij = self.expected_freqs[feature_type][:,:,i]
            else:
                freq_measures = [self.observed_freqs, self.expected_freqs]
                O, E = self.get_word_pair_measure(word_pair, feature_type, freq_measures)
                if O is None or E is None:
                    O_ij, E_ij = np.nan, np.nan
                else:
                    O_ij, E_ij = O[i], E[i]
            score += (O_ij - E_ij) ** 2 / E_ij
            # score[np.isnan(score)] = 0

        if visualize:
            self.visualize_score(score, feature_type, "Chi-square score")
        return score

    def log_likelihood_score(self, feature_type, word_pair=None, visualize=False):
        if word_pair is None:
            score = np.zeros_like(self.observed_freqs[feature_type][:,:,0])
        else:
            score = 0

        for i in np.arange(4):
            if word_pair is None:
                O_ij = self.observed_freqs[feature_type][:,:,i]
                E_ij = self.expected_freqs[feature_type][:,:,i]
            else:
                freq_measures = [self.observed_freqs, self.expected_freqs]
                O, E = self.get_word_pair_measure(word_pair, feature_type, freq_measures)
                if O is None or E is None:
                    O_ij, E_ij = np.nan, np.nan
                else:
                    O_ij, E_ij = O[i], E[i]

            s1 = O_ij * np.log(O_ij)
            try:
                s1[np.isnan(s1)] = 0
            except TypeError:
                s1 = 0 if np.isnan(s1) else s1
            s2 = O_ij * np.log(E_ij)
            try:
                s2[np.isnan(s2)] = 0
            except TypeError:
                s2 = 0 if np.isnan(s2) else s2
            score += s1 - s2
        score = 2 * score

        if visualize:
            self.visualize_score(score, feature_type, "Log-likelihood score", divergent=True)
        return score

    def dice_coefficient_score(self, feature_type, word_pair=None, visualize=False):
        if word_pair is None:
            O_11 = self.observed_freqs[feature_type][:,:,0]
            C_1 = self.marginal_freqs[feature_type][:,:,0]
            R_1 = self.marginal_freqs[feature_type][:,:,2]
        else:
            freq_measures = [self.observed_freqs, self.marginal_freqs]
            O, M = self.get_word_pair_measure(word_pair, feature_type, freq_measures)
            if O is None or M is None:
                O_11, C_1, R_1 = np.nan, np.nan, np.nan
            else:
                O_11, C_1, R_1 = O[0], M[0], M[2]

        score = 2 * O_11 / (C_1 + R_1)
        # score[np.isnan(score)] = 0

        if visualize:
            self.visualize_score(score, feature_type, "Dice coefficient score")
        return score

    def pmi_score(self, feature_type, word_pair=None, visualize=False):
        """Pointwise mutual information
        """
        if word_pair is None:
            O_11 = self.observed_freqs[feature_type][:,:,0]
            E_11 = self.expected_freqs[feature_type][:,:,0]
        else:
            freq_measures = [self.observed_freqs, self.expected_freqs]
            O, E = self.get_word_pair_measure(word_pair, feature_type, freq_measures)
            if O is None or E is None:
                O_11, E_11 = np.nan, np.nan
            else:
                O_11, E_11 = O[0], E[0]
        score = O_11 / E_11
        # score[np.isnan(score)] = 0

        if visualize:
            self.visualize_score(score, feature_type, "Pointwise mutual information score")
        return score

    def scp_score(self, feature_type, word_pair=None, visualize=False):
        """Symmetric conditional probability
        """
        if word_pair is None:
            O_11 = self.observed_freqs[feature_type][:,:,0]
            C_1 = self.marginal_freqs[feature_type][:,:,0]
            R_1 = self.marginal_freqs[feature_type][:,:,2]
        else:
            freq_measures = [self.observed_freqs, self.marginal_freqs]
            O, M = self.get_word_pair_measure(word_pair, feature_type, freq_measures)
            if O is None or M is None:
                O_11, C_1, R_1 = np.nan, np.nan, np.nan
            else:
                O_11, C_1, R_1 = O[0], M[0], M[2]
        score = O_11 ** 2 / (C_1 * R_1)
        # score[np.isnan(score)] = 0

        if visualize:
            self.visualize_score(score, feature_type, "Symmetric conditional probability score")
        return score

    def raw_frequency_score(self, feature_type, word_pair=None, visualize=False):
        if word_pair is None:
            O_11 = self.observed_freqs[feature_type][:,:,0]
        else:
            freq_measures = [self.observed_freqs]
            O = self.get_word_pair_measure(word_pair, feature_type, freq_measures)
            if O is None or O[0] is None:
                O_11 = np.nan
            else:
                O_11 = O[0]
        score = O_11
        # score[np.isnan(score)] = 0

        if visualize:
            self.visualize_score(score, feature_type, "Raw frequency score")
        return score

    def _save_data(self):
        save_data = {'class IDs': self.class_ids,
                     'expected frequencies': self.expected_freqs,
                     'feature vectors': self.feature_vectors,
                     'feature type values': self.feature_type_values,
                     'labels': self.labels,
                     'learning params': self.learning_params,
                     'marginal frequencies': self.marginal_freqs,
                     'observed frequencies': self.observed_freqs,
                     'test data indices': self.test_data_indices,
                     'training data size': self.training_data_size,
                     'training data indices': self.training_data_indices,
                     'truth data': self.truth_data,
                     'weights': self.weights,
                     'whiten': self.whiten,
                    }

        i = 1
        filename = self.filename
        while os.path.exists(filename):
            filename = self.filename[:-4] + "{}.npz".format(i)
            i += 1
        self.filename = filename

        np.savez(self.filename, **save_data)

    def _load_data(self):
        data  = np.load(self.filename)
        self.class_ids = data['class IDs']
        self.expected_freqs = data['expected frequencies'].item()
        self.feature_vectors = data['feature vectors']
        self.feature_type_values = data['feature type values'].item()
        self.labels = data['labels']
        self.learning_params = data['learning params']
        self.marginal_freqs = data['marginal frequencies'].item()
        self.observed_freqs = data['observed frequencies'].item()
        self.test_data_indices = data['test data indices']
        self.training_data_size = data['training data size']
        self.training_data_indices = data['training data indices']
        self.truth_data = data['truth data']
        self.weights = data['weights']
        self.whiten = data['whiten']

    def _parse_corpus(self, corpus):
        # Pre-allocate corpus/feature dicts
        self.corpus = {'full': None,
                       'lemmata': None,
                       'POS tags': None,
                       }

        # Grab corpus
        if corpus is not None:
            self.corpus['full'] = corpus
        else:
            self.dh = DataHandler()
            self.corpus['full'] = self.dh.corpus
        self._lemmatize_corpus()
        self._pos_tag_corpus()

        # Pre-allocate frequency values
        self.observed_freqs = {'lemmata': np.zeros((len(self.lemmata), len(self.lemmata),4)),
                               'POS tags': np.zeros((len(self.pos_tags), len(self.pos_tags),4))
                               }

        self.expected_freqs = {'lemmata': np.zeros((len(self.lemmata), len(self.lemmata),4)),
                               'POS tags': np.zeros((len(self.pos_tags), len(self.pos_tags),4))
                               }

        self.marginal_freqs = {'lemmata': np.zeros((len(self.lemmata), len(self.lemmata),4)),
                               'POS tags': np.zeros((len(self.pos_tags), len(self.pos_tags),4))
                               }

    def _lemmatize_corpus(self):
        doc = self.nlp(unicode(self.corpus['full']))
        self.corpus['lemmata'] = [None] * len(doc)
        for i, token in enumerate(doc):
            self.corpus['lemmata'][i] = token.lemma_
        self.lemmata = list(set(self.corpus['lemmata']))
        self.feature_type_values['lemmata'] = self.lemmata

    def _pos_tag_corpus(self):
        doc = self.nlp(unicode(self.corpus['full']))
        self.corpus['POS tags'] = [None] * len(doc)
        for i, token in enumerate(doc):
            self.corpus['POS tags'][i] = token.tag_
        self.pos_tags = list(set(self.corpus['POS tags']))
        self.feature_type_values['POS tags'] = self.pos_tags

    def _count_word_pairs(self, word_pairs, feature_type):
        """
        Takes in a lemmatized word pair (u,v) and compares them to each lemma
        pair (U,V) in pre-trained lemmata.

        Counts one of four values:
        * 0 if U = u and V = v
        * 1 if U = u and V ~= v
        * 2 if U ~= u and V = v
        * 3 if U ~= u and V ~= v
        """

        #<>TODO: figure out what to do when u_i == v_i
        for u_i, u in enumerate(self.feature_type_values[feature_type]):
            for v_i, v in enumerate(self.feature_type_values[feature_type]):
                for word_pair in word_pairs:
                    l1, l2 = word_pair
                    if u == l1 and v == l2:
                        self.observed_freqs[feature_type][u_i, v_i, 0] += 1
                    elif u == l1 and v != l2:
                        self.observed_freqs[feature_type][u_i, v_i, 1] += 1
                    elif u != l1 and v == l2:
                        self.observed_freqs[feature_type][u_i, v_i, 2] += 1
                    elif u != l1 and v != l2:
                        self.observed_freqs[feature_type][u_i, v_i, 3] += 1

    def _create_contingency_tables(self, feature_type=None):
        if feature_type is None:
            types = self.feature_type_values.keys()
        else:
            types = [feature_type]

        for feature_type in types:
            # Loop over all word pair lemmata in the corpus
            word_pairs = list(zip(self.corpus[feature_type][:-1],
                                  self.corpus[feature_type][1:]))
            N = len(word_pairs)
            self._count_word_pairs(word_pairs, feature_type)

            # Define observed frequency marginals
            C_1 = (self.observed_freqs[feature_type][:, :, 0]
                   + self.observed_freqs[feature_type][:, :, 2])
            C_2 = (self.observed_freqs[feature_type][:, :, 1]
                   + self.observed_freqs[feature_type][:, :, 3])
            R_1 = (self.observed_freqs[feature_type][:, :, 0]
                   + self.observed_freqs[feature_type][:, :, 1])
            R_2 = (self.observed_freqs[feature_type][:, :, 2]
                   + self.observed_freqs[feature_type][:, :, 3])
            self.marginal_freqs[feature_type] = np.dstack((C_1, C_2, R_1, R_2))
            N = C_1[0,0] + C_2[0,0]

            E11 = R_1 * C_1 / N
            E12 = R_1 * C_2 / N
            E21 = R_2 * C_1 / N
            E22 = R_2 * C_2 / N
            self.expected_freqs[feature_type] = np.dstack((E11, E12, E21, E22))

    def _parse_truth_data(self, split_pct=0.5):
        # Grab token data
        try:
            self.dh
        except AttributeError:
            self.dh = DataHandler()

        true_tokens = {'sierra': self.dh.df["Sierra's Tokens"].tolist(),
                       'jeremy': self.dh.df["Jeremy's Tokens"].tolist(),
                      }
        num_datasets = 2

        # Ignore NaN lines
        true_tokens['sierra'] = [d for d in true_tokens['sierra']
                                 if not isinstance(d, float)]
        true_tokens['jeremy'] = [d for d in true_tokens['jeremy']
                                 if not isinstance(d, float)]

        # Combine into the full dataset
        true_tokens['full'] = true_tokens['sierra'] + true_tokens['jeremy']
        self.true_tokens = true_tokens

        # Get sequential word pairs with group labels from corpus and tokens
        single_word_tokens = self.dh.get_single_word_tokens()
        num_word_pairs = len(single_word_tokens) - 1
        data_size = num_word_pairs * num_datasets
        word_pair_groups = [None] * num_word_pairs * (len(true_tokens) - 1)

        # Classify each word pair in training set as grouped or not grouped
        for d, dataset in enumerate(['sierra', 'jeremy']):
            k = 0
            token_list = []
            for i, u in enumerate(single_word_tokens[:-1]):
                v = single_word_tokens[i + 1]

                # Pop off a new multi word tokenization as necessary
                if len(token_list) == 0:
                    if k > len(true_tokens[dataset]) - 1:
                        token_list = []
                    else:
                        token_list = true_tokens[dataset][k].split(' ')

                    # Clean up the token list
                    token_list = [t for t in token_list if len(t) > 0]

                    k += 1

                # Compare u and v to the token list (stop if broken comparison)
                list_u = token_list[0]
                if u.upper() not in list_u.upper():
                    logging.exception("i={}, u = {}, list_u = {}"
                                      .format(i, u, list_u))
                    raise ValueError
                token_list = token_list[1:]
                try:
                    is_grouped = v in token_list[0]
                except IndexError:
                    is_grouped = False

                index = i + d * num_word_pairs
                word_pair_groups[index] = (u, v, is_grouped)


        # Figure out training data size and indices
        self.training_data_size = int(round(data_size * split_pct))
        self.training_data_indices = np.random.choice(data_size, self.training_data_size, replace=False)
        self.test_data_indices = np.array([i for i in np.arange(data_size)
                                           if i not in self.training_data_indices])

        # Create one variable to hold all training and test data
        self.truth_data = {'full': word_pair_groups}
        self.truth_data['training'] = [word_pair_groups[i] for i in self.training_data_indices]
        self.truth_data['test'] = [word_pair_groups[i] for i in self.test_data_indices],

    def _generate_features_and_labels(self):
        """Current datasets are 'jeremy' and 'sierra'
        """
        self.num_features = len(self.feature_type_values) * len(Scorer.score_functions)

        word_pair_groups = self.truth_data['full']
        self.feature_vectors = np.ones((len(word_pair_groups),
                                         self.num_features + 1))
        self.labels = np.empty((len(word_pair_groups), self.num_classes))
        self.class_ids = np.empty(len(word_pair_groups))
        for i, word_pair_group in enumerate(word_pair_groups):
            word_pair = word_pair_group[:2]

            score = self.score_words(word_pair)
            self.feature_vectors[i][1:] = self.score_vector_from_scores(score)

            #<>TODO: extend for multi-class scenario
            if word_pair_group[-1]:
                self.labels[i] = [1, 0]
                self.class_ids[i] = 1
            else:
                self.labels[i] = [0, 1]
                self.class_ids[i] = 0

        # Whiten by centering on mean and dividing by standard deviation
        if self.whiten:
            # import pdb; pdb.set_trace()  # breakpoint a943760a //

            self.feature_vectors[np.isinf(self.feature_vectors)] = np.nan
            self.feature_vectors[:, 1:] -= np.nanmean(self.feature_vectors[:, 1:], axis=0)
            self.feature_vectors[:, 1:] /= np.nanstd(self.feature_vectors[:, 1:], axis=0)

    def _learn_weights(self):
        # Set up training data, incl. column of 1's for bias feature
        n = self.training_data_size
        feature_vectors = self.feature_vectors[self.training_data_indices]
        labels = self.labels[self.training_data_indices]

        #<>TODO: Cheap hack for now - more principled way to fix this?
        feature_vectors[np.isinf(feature_vectors)] = 0

        # Randomly initialize parameters (+1 for bias term)
        weights = np.random.normal(0, 1, (self.num_features + 1, self.num_classes))
        grad = np.ones((self.num_features + 1, self.num_classes))

        iter_ = 0
        # STEP = 50
        # import pdb; pdb.set_trace()  # breakpoint 3006a4b7 //
        while iter_ < self.max_iter and all(self.min_step < np.abs(grad[grad>0])):

            # Find the hypothesis
            # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
            M = weights.T .dot (feature_vectors.T)
            b = np.nanmax(M)  # To deal with overflow
            log_hyp = M - logsumexp(M,axis=0)
            hypothesis = np.exp(log_hyp)
            hypothesis[np.isnan(hypothesis)] = 0.5  # <>TODO: remove cheap hack!

            # Find the gradient at the given parameters
            grad = (-1 / n) * (feature_vectors.T .dot (labels - hypothesis.T))
            grad += self.weight_decay * weights
            grad[np.isnan(grad)] = 0
            # logging.info("{}, {}".format(iter_, np.linalg.norm(grad)))
            # if np.mod(iter_, STEP) == 0:
                # import pdb; pdb.set_trace()  # breakpoint 3006a4b7 //
        
            # Take one steps in gradient descent on feature space
            weights = weights - self.step_size * grad
            iter_ += 1

        self.weights = weights
        logging.info('Finished after {} iterations and {} grad norm'.format(iter_, np.linalg.norm(grad)))

    score_functions = OrderedDict((('t-score', t_score),
                                   ('z-score', z_score),
                                   ('chi-square', chi_square_score),
                                   ('log-likelihood', log_likelihood_score),
                                   ('dice coefficient', dice_coefficient_score),
                                   ('pointwise mutual information', pmi_score),
                                   ('symmetric conditional probability', scp_score),
                                   ('raw frequency', raw_frequency_score),
                                   ))




def test_scorer():
    # scorer = Scorer("this that that that cat")
    # scorer = Scorer("The red robot is near the chair.")
    # scorer = Scorer("Oh my gosh the oh my goshes are goshing")
    scorer = Scorer()
    # scorer.view_truth_data()
    scorer.score_words(visualize=True, feature_type=None)
    scorer.test_accuracy()
    # print scorer.classify_word_pair(("is", "moving"))

    # accuracies = []
    # for i in range(10):
    #     accuracies.append(scorer.test_accuracy())

    # print accuracies
    # print scorer.truth_data

    # print scorer.t_score('lemmata', ('is','mayonnaise'))
    
    # scorer.score_words(visualize=True, feature_type=None)
    # scores = scorer.score_words(word_pair=('is', 'near'))
    # print scores
    # print scorer.score_vector_from_scores(scores)

    plt.show()




def test_tokenizer(document='', max_distance=3):
    if len(document) < 1:
        document = "Roy is behind you."

    tokenizer = Tokenizer()
    # tokenized_document = tokenizer.tokenize(document)
    # print tokenized_document
    tokenizer.test_accuracy()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    test_scorer()
    # test_tokenizer()
    # test_tokenizer("Roy is around the corner, inside the kitchen, near the fridge.", max_distance=5)
