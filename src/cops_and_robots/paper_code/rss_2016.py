#!/usr/bin/env python
"""Collection for the 2016 RSS workshop paper.
"""
from __future__ import division
__author__ = "Nick Sweet"
__copyright__ = "Copyright 2016, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import logging
from itertools import product
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import spacy.en


from cops_and_robots.fusion.softmax import binary_speed_model
from cops_and_robots.human_tools.statement_template import StatementTemplate, get_all_statements


def speed_example_binary():
    """
    Problems - how do you account for double counting? 
    How do you arrive at all joint likelihoods?
    How do you find the probabilities of each joint likelihood given a tokenization?
    """

    # Load two binary softmax models
    bsm = binary_speed_model()
    models = OrderedDict({'Slow': bsm.binary_models['Slow'],
                          'Fast': bsm.binary_models['Fast'],
                         })
    keys = models.keys()

    # Generate joint likelihood of each of four possibilities
    models[keys[0]]._define_state()
    x = np.squeeze(models[keys[0]].state)
    xlim = [x.min(), x.max()]
    joint_likelihoods = {}
    truth_values = product([True,False], repeat=len(models))
    for truths in truth_values:
        str_ = ''
        likelihoods = []

        # Find individual likelihoods P(Dk = i|Xk)
        for i, truth in enumerate(truths):
            str_ += keys[i] + '_' + str(truths[i]) + ' + '
            class_ = keys[i]
            if not truth:
                class_ = 'Not ' + class_
            likelihood = models[keys[i]].probability(class_=class_)
            del models[keys[i]].probs
            likelihoods.append(likelihood)
        str_ = str_[:-3]

        # Combine them into joint likelihoods P(Dk = i1|Xk)P(Dk = i1|Xk)
        joint_likelihood = np.ones(x.size)
        for likelihood in likelihoods:
            joint_likelihood *= likelihood
        joint_likelihoods[str_] = joint_likelihood

    # Find the marginalized likelihood with uninformative tokenization
    weights1 = [0.25] * 4  # Uniform joint model likelihoods P(Dk = 1, 2, 3, 4|Tk)
    normalizers = [0.25] * 4 # 
    average_likelihood = np.zeros_like(joint_likelihood)
    i = 0
    for _, joint_likelihood in joint_likelihoods.iteritems():
        normalizer = 1
        # for j, x_val in enumerate(x):
        #     normalizer += joint_likelihood[j] * x_val
        average_likelihood += weights1[i] * joint_likelihood / normalizers[i]
        i += 1

    # Find the marginalized likelihood with uninformative tokenization
    weights2 = [0.2, 0.3, 0.3, 0.2]  # weighted towards fast true
    normalizers = [0.25] * 4 # 
    weighted_average_likelihood = np.zeros_like(joint_likelihood)
    i = 0
    for _, joint_likelihood in joint_likelihoods.iteritems():
        normalizer = 1
        # for j, x_val in enumerate(x):
        #     normalizer += joint_likelihood[j] * x_val
        weighted_average_likelihood += weights2[i] * joint_likelihood / normalizers[i]
        i += 1

    # Find the marginalized likelihood with uninformative tokenization
    weights3 = [0.0, 0.5, 0.5, 0.0]  # weighted towards fast true
    normalizers = [0.25] * 4 # 
    weighted_average_likelihood2 = np.zeros_like(joint_likelihood)
    i = 0
    for _, joint_likelihood in joint_likelihoods.iteritems():
        normalizer = 1
        # for j, x_val in enumerate(x):
        #     normalizer += joint_likelihood[j] * x_val
        weighted_average_likelihood2 += weights3[i] * joint_likelihood / normalizers[i]
        i += 1


    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(5,2,1)
    models['Slow'].plot(plot_dominant_classes=False, fig=fig, ax=ax, show_plot=False)
    ax.set_xlim(xlim)

    ax = fig.add_subplot(5,2,2)
    models['Fast'].plot(plot_dominant_classes=False, fig=fig, ax=ax, show_plot=False)
    ax.set_xlim(xlim)

    i = 3
    for joint_model_name, joint_likelihood in joint_likelihoods.iteritems():
        ax = fig.add_subplot(5,2,i)
        ax.plot(x, joint_likelihood)
        ax.fill_between(x, 0, joint_likelihood, alpha=0.3)
        ax.set_title(joint_model_name)
        ax.set_xlim(xlim)

        i += 1

    ax = fig.add_subplot(5,2,i)
    ax.plot(x, average_likelihood)
    ax.fill_between(x, 0, average_likelihood, alpha=0.3)
    str_ = "Average likelihood (weights: {})".format(weights1)
    ax.set_title(str_)
    ax.set_xlim(xlim)


    i += 1
    ax = fig.add_subplot(5,2,i)
    ax.plot(x, weighted_average_likelihood)
    ax.fill_between(x, 0, weighted_average_likelihood, alpha=0.3)
    str_ = "Average likelihood (weights: {})".format(weights2)
    ax.set_title(str_)
    ax.set_xlim(xlim)


    i += 1
    ax = fig.add_subplot(5,2,i)
    ax.plot(x, weighted_average_likelihood2)
    ax.fill_between(x, 0, weighted_average_likelihood2, alpha=0.3)
    str_ = "Average likelihood (weights: {})".format(weights3)
    ax.set_title(str_)
    ax.set_xlim(xlim)


    plt.tight_layout()
    plt.show()


def speed_example_categorical():
        # Load two binary softmax models
    bsm = binary_speed_model()
    models = OrderedDict({'Slow': bsm.binary_models['Slow'],
                          'Fast': bsm.binary_models['Fast'],
                         })
    keys = models.keys()

    # Weight each of the four statements
    models[keys[0]]._define_state()
    x = np.squeeze(models[keys[0]].state)
    xlim = [x.min(), x.max()]

    weight_sets = np.array([[0.25, 0.25, 0.25, 0.25],
                            [0.5, 0, 0.5, 0],
                            [0, 0.5, 0, 0.5],
                            [0, 0, 1.0, 0],
                            ])
    weight_sets = (weight_sets.T / weight_sets.sum(axis=1)).T

    weighted_likelihoods = []
    for weights in weight_sets:
        weighted_likelihood = np.zeros(x.size)
        i = 0
        class_names = []
        for model_name, model in models.iteritems():
            for truth in [True, False]:
                class_ = model_name
                if not truth:
                    class_ = 'Not ' + class_
                likelihood = model.probability(class_=class_)

                likelihood = model.probability(class_=class_)
                weighted_likelihood += weights[i] * likelihood
                i += 1
                class_names.append(class_)
        weighted_likelihoods.append(weighted_likelihood)


    fig = plt.figure(figsize=(12,8))
    n = weight_sets.shape[0] + 1
    ax = plt.subplot2grid((n,2), (0,0))
    del models['Slow'].probs
    models['Slow'].plot(plot_dominant_classes=False, fig=fig, ax=ax, show_plot=False, title='')
    ax.set_xlim(xlim)

    ax = plt.subplot2grid((n,2), (0,1))
    del models['Fast'].probs
    models['Fast'].plot(plot_dominant_classes=False, fig=fig, ax=ax, show_plot=False, title='')
    ax.set_xlim(xlim)

    for i, weights in enumerate(weight_sets):
        ax = plt.subplot2grid((n,2), (1+i,0), colspan=2)
        ax.plot(x, weighted_likelihoods[i])
        ax.fill_between(x, 0, weighted_likelihoods[i], alpha=0.3)
        str_ = "Average likelihood (weights: {} for {})".format(weights,class_names)
        ax.set_title(str_)
        ax.set_xlim(xlim)
        ax.set_ylim([0, 1.2 * weighted_likelihoods[i].max()])

    plt.tight_layout()
    plt.show()


def word_vector_projection_example():

    # Add custom arrow class from 
    # http://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)

    def plot_vecs(vector_list, ax):
        colors = ['r','b','lime']
        max_, min_ = -1000, 1000
        for i, vectors in enumerate(vector_list):
            for v in vectors:
                if len(v) == 2:
                    vec = [0, v[0]], [0, v[1]], [0, 0]
                else:
                    vec = [0, v[0]], [0, v[1]], [0, v[2]]
                a = Arrow3D(*vec, mutation_scale=10, lw=2, arrowstyle="-|>", color=colors[i])
                ax.add_artist(a)

            min_ = vectors.min() if min_ > vectors.min() else min_
            max_ = vectors.max() if max_ < vectors.max() else max_

        ax.set_xlim([min_, max_])
        ax.set_ylim([min_, max_])
        ax.set_zlim([min_, max_])

    n_random_vecs = 100
    n_known_vecs = 2
    n_input_vecs = 1
    random_3d_vecs = np.random.normal(0, 1, size=(n_random_vecs, 3))
    known_3d_vecs = np.random.normal(1, 0.25, size=(n_known_vecs, 3))
    input_3d_vecs = np.random.normal(1, 0.25, size=(n_input_vecs, 3))

    vector_list = [random_3d_vecs, known_3d_vecs, input_3d_vecs]


    # Compare vectors
    cos_similarities = np.zeros((known_3d_vecs.shape[0], input_3d_vecs.shape[0]))
    for i, known_vec in enumerate(known_3d_vecs):
        for j, input_vec in enumerate(input_3d_vecs):
            known_norm = np.linalg.norm(known_vec)
            input_norm = np.linalg.norm(input_vec)
            cos_similarities[i] = known_vec.T .dot (input_vec) / (known_norm * input_norm)

    print "Initial similarities:"
    print cos_similarities


    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(121, projection="3d")
    plot_vecs(vector_list, ax)
    ax.set_title('Original Vectors')

    # Perform QR decomposition on known vectors and reproject in new basis
    Q, R = np.linalg.qr(known_3d_vecs.T)
    rank = np.linalg.matrix_rank(R)

    # print "Q shape and rank: \n", Q.shape, np.linalg.matrix_rank(Q)
    # print "R shape and rank: \n", R.shape, np.linalg.matrix_rank(R)
    # print "Known vectors A: \n", known_3d_vecs, "\n"
    # print "QR - A: \n", Q .dot (R) - known_3d_vecs.T, "\n"
    # print "QQ^T: \n", Q .dot (Q.T), "\n"
    # print "R - Q^TA: \n", R - Q.T .dot (known_3d_vecs.T), "\n"

    # print "random shape and rank: \n", random_3d_vecs.shape, np.linalg.matrix_rank(random_3d_vecs)
    # print "input shape and rank: \n", input_3d_vecs.shape, np.linalg.matrix_rank(input_3d_vecs)
    # print "known shape and rank: \n", known_3d_vecs.shape, np.linalg.matrix_rank(known_3d_vecs)


    random_3d_vecs = (Q.T .dot (random_3d_vecs.T)).T
    known_3d_vecs = (Q.T .dot (known_3d_vecs.T)).T
    input_3d_vecs = (Q.T .dot (input_3d_vecs.T)).T
    # print "random shape and rank: \n", random_3d_vecs.shape, np.linalg.matrix_rank(random_3d_vecs)
    # print "input shape and rank: \n", input_3d_vecs.shape, np.linalg.matrix_rank(input_3d_vecs)
    # print "known shape and rank: \n", known_3d_vecs.shape, np.linalg.matrix_rank(known_3d_vecs)

    vector_list = [random_3d_vecs, known_3d_vecs, input_3d_vecs]
    ax = fig.add_subplot(122, projection="3d")
    plot_vecs(vector_list, ax)
    ax.set_title('Gram-schmidt Orthogonalized Vectors')

    # Compare vectors
    new_cos_similarities = np.zeros((known_3d_vecs.shape[0], input_3d_vecs.shape[0]))
    for i, known_vec in enumerate(known_3d_vecs):
        for j, input_vec in enumerate(input_3d_vecs):
            known_norm = np.linalg.norm(known_vec)
            input_norm = np.linalg.norm(input_vec)
            new_cos_similarities[i] = known_vec.T .dot (input_vec) / (known_norm * input_norm)

    print "New similarities:"
    print new_cos_similarities

    print "Old - new:"
    print cos_similarities - new_cos_similarities


    plt.show()


def word_vector_generation():
    from numpy import dot
    from numpy.linalg import norm
    import copy

    statement_template = StatementTemplate(add_more_relations=True)
    
    logging.info("Loading SpaCy...")
    nlp = spacy.en.English(parser=False)
    logging.info("Done!")

    # components = ['left','right','front','back','near']
    input_words = ['desk', 'bookshelf', 'table', 'kitchen']
    components = ['desk','chair','fridge','stove','bookcase']
    context_words = {'desk': {'table', 'surface', 'cubicle', 'counter'},
                     'chair': {'beanbag', 'seat', 'stool', 'couch'},
                     'fridge': {'freezer', 'kitchen', 'icebox', 'refrigerator'},
                     'stove': {'oven', 'pan', 'electic', 'couch'},
                     'bookcase': {'beanbag', 'seat', 'stool', 'couch'},
                    }
    # components = statement_template.components['grounding']['object']
    vectors = {'input': {},
               'template': {},
               'context': {},
               }

    similarities = {c: {} for c in input_words}

    sim = lambda word1, word2: word1.similarity(word2)
    vocabulary = list({w for w in nlp.vocab if w.has_vector and w.orth_.islower()})

    for component in components:
        component_tok = nlp(unicode(component))
        # component_tok = component_tok[-1]
        vectors['template'][component] = component_tok.vector * component_tok.vector_norm

        for input_word in input_words:
            input_word_tok = nlp(unicode(input_word))
            vectors['input'][input_word] = input_word_tok.vector * input_word_tok.vector_norm
            similarities[input_word][component] = component_tok.similarity(input_word_tok)

        vocabulary.sort(key=lambda w: sim(component_tok, w))
        vocabulary.reverse()

        # print("\nTop 10 most similar words to '{}':".format(component_tok))
        print("\nFor the word '{}', find all ranked contexts:".format(component_tok))
        context_vecs = {}
        # context_vocab = copy.deepcopy(vocabulary[:1000])
        context_vocab = vocabulary[:1000]
        for other_component in components:
            if other_component == component:
                continue
            other_component_tok = nlp(unicode(other_component))
            other_component_tok = other_component_tok[0]

            # print word.orth_.encode('utf8'), component_tok.similarity(word)
            # print word.orth_.encode('utf8')
            # print word
            # context_vecs[word.orth_.encode('ascii')] = word.vector * word.vector_norm

            print("\nTop 10 most similar words to '{}':".format(other_component_tok.orth_.encode('ascii')))
            context_vocab.sort(key=lambda w: sim(other_component_tok, w))
            context_vocab.reverse()
            for v in context_vocab[:10]:
                print v.orth_.encode('utf8'), component_tok.similarity(v)
            # pass

        vectors['context'][component] = context_vecs

    # print similarities
    # print vectors
    # from scipy.io import savemat
    # output = {'vectors': vectors,
    #           'similarities': similarities,
    #           }
    # savemat("output.mat", output)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    speed_example_binary()
    # speed_example_categorical()
    # word_vector_projection_example()
    # word_vector_generation()

