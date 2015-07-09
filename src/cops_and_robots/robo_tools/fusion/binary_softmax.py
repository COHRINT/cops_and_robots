#!/usr/bin/env python
"""Provides creation, maniputlation and plotting of Softmax distributions.

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
import copy

from cops_and_robots.robo_tools.fusion.softmax import Softmax, speed_model,\
    pentagon_model, distance_space_model, intrinsic_space_model, \
    make_regular_2D_poly


class BinarySoftmax(Softmax):
    """A collection of Binary versions of Softmax distributions.

    While the Softmax class can take m>=2 class labels to create m mutually
    exclusive and exhaustive distributions,the BinarySoftmax class takes
    m>=2 class labels to create m sets of 2 distributions. Each set contains
    one of the previous Softmax distributions and its complement.

    For example, given a one-dimensional speed model with m=3 class labels
    'stopped', 'slowly', and 'quickly', the new BinarySoftmax model creates
    'stopped' and 'not stopped', 'slowly' and 'not slowly', and 'quickly' and
    'not quickly'. Each set is mutually exclusive and exhaustive, but there is
    no longer a dependency between the original labels.

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

    def __init__(self, softmax_model):
        super(BinarySoftmax, self).__init__(weights=np.zeros((2, 2)),
                                            biases=np.zeros(2),
                                            class_labels=['Null', 'NaC'],
                                            )
        self.softmax_model = softmax_model

        # Remove unwanted bits of the softmax superclass
        del self.weights
        del self.biases
        del self.class_cmaps
        del self.class_colors
        del self.class_labels
        del self.normals
        del self.num_classes
        del self.num_params
        del self.offsets
        del self.poly
        del self.probs

        self.categorical_to_binary()

    def categorical_to_binary(self):
        """Transforms a m>2 class softmax model to multiple binary models.
        """
        self.binary_models = {}
        for label in self.softmax_model.class_labels:
            new_softmax = copy.deepcopy(self.softmax_model)
            new_softmax.other_weights = []

            for i, new_label in enumerate(new_softmax.class_labels):
                if new_label != label:
                    new_label = 'not ' + label
                    new_softmax.other_weights.append(
                        self.softmax_model.weights_by_label(label))
                    new_softmax.class_labels[i] = new_label.title()

            new_softmax.combine_mms()

            self.binary_models[label] = new_softmax

    def probs_at_state(self, state, label):
        if 'Not ' in label:
            not_label = label
            label = label.replace('Not ', '')
            p = self.binary_models[label].probs_at_state(state, not_label)
        else:
            p = self.binary_models[label].probs_at_state(state, label)
        return p


def binary_speed_model():
    sm = speed_model()
    bsm = BinarySoftmax(sm)
    return bsm


def binary_distance_space_model(poly=None):
    dsm = distance_space_model(poly)
    bdsm = BinarySoftmax(dsm)
    return bdsm


def binary_intrinsic_space_model(poly=None):
    ism = intrinsic_space_model(poly)
    bism = BinarySoftmax(ism)
    del bism.binary_models['Inside']
    return bism


if __name__ == '__main__':

    # bsm = binary_speed_model()
    # bsm.binary_models['Slow'].plot()
    # <>TODO: Ensure the color of the 'not' is always grey
    # pent = pentagon_model()
    # bpent = BinarySoftmax(pent)

    # bpent.binary_models['Heliport Facade'].plot()

    bdsm = binary_distance_space_model()
    bdsm.binary_models['Outside'].plot()

    # bism = binary_intrinsic_space_model()
    # bism.binary_models['Right'].plot()