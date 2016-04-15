from __future__ import division
import logging
import numpy as np
import copy


from cops_and_robots.fusion.softmax import Softmax

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

    def __init__(self, softmax_model, allowed_relations=None, bounds=None):
        super(BinarySoftmax, self).__init__(weights=np.zeros((2, 2)),
                                            biases=np.zeros(2),
                                            labels=['Null', 'NaC'],
                                            bounds=bounds,
                                            )
        self.softmax_model = softmax_model

        # Remove unwanted bits of the softmax superclass
        del self.weights
        del self.biases
        del self.classes
        del self.class_cmaps
        del self.class_colors
        del self.class_labels
        del self.normals
        del self.num_classes
        del self.num_params
        del self.offsets
        del self.poly

        self.categorical_to_binary()

    def __str__(self):
        str_ = ('Binary softmax model with submodels: {}'
            .format( self.binary_models.keys()))
        return str_

    def categorical_to_binary(self):
        """Transforms a m>2 class softmax model to multiple binary models.
        """
        self.binary_models = {}

        # Create new binary softmax model for each class
        for class_label in self.softmax_model.class_labels:
            new_softmax = copy.deepcopy(self.softmax_model)
            
            # If MMS model use subclass labels
            if hasattr(new_softmax, 'subclasses'):
                new_softmax.labels = []
                for l in new_softmax.subclass_labels:
                    j = l.find('__')
                    if j > -1:
                        l = l[:j] 
                    new_softmax.labels.append(l)
                del new_softmax.subclasses
            else:
                new_softmax.labels = new_softmax.class_labels
            del new_softmax.classes
            
            if hasattr(new_softmax,'probs'):
                del new_softmax.probs
            if hasattr(new_softmax,'subclass_probs'):
                del new_softmax.subclass_probs

            for i, new_label in enumerate(new_softmax.labels):
                if new_label != class_label:
                    new_label = 'not ' + class_label
                    new_softmax.labels[i] = new_label.title()

            new_softmax.num_classes = len(new_softmax.labels)
            new_softmax._define_classes(new_softmax.labels)

            self.binary_models[class_label] = new_softmax

    def probability(self, state=None, class_=None):
        # if class_ == None:
        #     class_ = 
        if 'Not ' in class_:
            not_label = class_
            label = class_.replace('Not ', '')
            p = self.binary_models[label].probability(state, not_label)
        else:
            label = class_
            p = self.binary_models[label].probability(state, label)
        return p

    def trim_categories(self):
        pass
    # <>TODO: Subclass dict to BinaryDict, allowing us to call any class from
    # a binary MMS model
    # @property
    # def classes(self):
    #     for binary_model in self.binary_models:
    #         try:
    #             class_ = binary_model[key]
    #             return class_
    #         except KeyError:
    #             logging.debug('No class {} in {}.'.format(key, binary_model))
    #         except e:
    #             raise e