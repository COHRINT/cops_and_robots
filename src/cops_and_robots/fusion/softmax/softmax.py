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

from shapely.geometry import box, Polygon

from descartes.patch import PolygonPatch

import warnings  # To suppress nolabel warnings
warnings.filterwarnings("ignore", message=".*cannot be automatically added.*")


class Softmax(object):
    """Generate a softmax distribution from weights or class boundaries.

    The relative magnitude of weights defines the slope of each softmax
    class probability, and the hyperplanes at which one softmax class
    crosses another represents the boundary between those two classes.
    Boundaries between two dominant classes (those with greatest
    probability for a given region) are called critical boundaries, and
    their normals are the only ones that need to be specified.

    If a space has non-dominant classes (and non-critical boundaries), then
    the weights are slack variables that can take on any values so long as
    the normals of ``all`` class boundaries sum up to the 0 vector.

    Since a class represents a probability distribution, the probabilities of
    all classes at any state must sum to one.

    Note
    ----
    Either the weights and biases, normals and offsets, or poly must be
    specified. Their effects are mutually exclusive, and take precedence in the
    above order.

    .. image:: img/classes_Softmax.png

    Parameters
    ----------
    weights : array_like, semi-optional
        Weights must be specified as an `M` by `N` multidimensional array, with
        each sub-array (akin to a matrix row) representing a 1-dimensional
        array (akin to an `N`-dimensional vector) of weights for each of the
        `M` classes. `N` must be the same size as the state vector.
    biases : array_like, semi-optional
        Provided with weights, biases must be specified as a 1-dimensional
        array containing the bias of all `M` classes.

        Defaults to an array of `M` zeros.
    normals : array_like, semi-optional
        Normals must be specified as an `M` by `N` multidimensional array, with
        each sub-array (akin to a matrix row) representing a 1-dimensional
        array (akin to an `N`-dimensional vector) of coefficients for each of
        the normal vectors representing `M(M + 1) / 2` class boundaries.
        `N` must be the same size as the state vector.
    offsets : array_like, semi-optional
        Provided with normals, offsets must be specified as a 1-dimensional
        array containing the constant offset of all `M` classes.

        Defaults to an array of `M` zeros.
    poly : shapely.Polygon, semi-optional
        A polygon object from `Shapely`. Used to create normals.
    steepness : array_like, optional
        Steepness must be specified as a 1-dimensional array containing scalar
        values that modify the shape of `M` classes by linearly scaling their
        weights.
    rotation : float, optional
        For a two-dimensional state space, `rotation` is the angular rotation
        in radians of the entire state space. It is not defined for higher- or
        lower-order state spaces.
    state_spec : str, optional
        A string specified as defined in :func:`~softmax.Softmax.define_state`.
    bounds : array_like, optional
        Distribution boundaries, in meters, as [xmin, ymin, xmax, ymax].
        Defaults to [-5, -5, 5, 5].

        Also applies to non-spatial bounds. For instance, if the state space is
        'x y x_dot y_dot', bounds should be [xmin, ymin, x_dot_min, y_dot_min,
        xmax, ymax, x_dot_max, y_dot_max].
    resolution : float, optional
        Grid resolution over which the state space is defined. Defaults to 0.1.
    labels : list of str, optional
        If given, assigns labels to each subclass and allows for consistent
        colorization between each superclass (i.e. all classes with the same
        label will have the same color).

    Note
    ----
    The Softmax function takes the form:

    .. math::

        P(D=j \\vert x) = \\frac{ e^{ \\mathbf{x}^T \\mathbf{w}_j }}
            {\\sum_{k=1}^{M} e^{\\mathbf{x}^T \\mathbf{w}_k} }

    Where `x` is the state vector.

    The log-odds boundaries between two classes form linear hyperplanes:

    .. math::

        \\log(\\frac{ P(D=j \\vert x) }{ P(D=k \\vert x)}) = \\mathbf{x}^T
            (\\mathbf{w}_j - \\mathbf{w}_k)

    """
    class_cmaps = ['Greys', 'Reds', 'Purples', 'Oranges', 'Greens', 'Blues',
                   'RdPu']
    class_colors = ['grey', 'red', 'purple', 'orange', 'green', 'blue', 'pink']

    # Load methods from external files
    from _visualization import (plot,
                                _plot_probs,
                                _plot_probs_1D,
                                _plot_probs_2D,
                                _plot_probs_3D,
                                _plot_dominant_classes,
                                _plot_dominant_classes_1D,
                                _plot_dominant_classes_2D,
                                _plot_dominant_classes_3D,
                                )


    def __init__(self, weights=None, biases=None, normals=None, offsets=None,
                 poly=None, steepness=None, rotation=None, state_spec='x y',
                 bounds=None, resolution=0.1, labels=None,
                 auto_combine_mms=True, tol=10 ** -3):

        try:
            assert 1 == sum([1 for a in [weights, normals, poly]
                             if a is not None])
        except AssertionError, e:
            logging.exception('One of weights, normals or poly must be '
                              'specified - choose only one!')
            raise e

        # Define base object attributes
        self.weights = weights
        self.biases = biases
        self.normals = normals
        self.offsets = offsets
        self.poly = poly
        self.steepness = steepness
        if bounds is None:
            bounds = [-5, -5, 5, 5]
        self.bounds = bounds
        self.state_spec = state_spec
        self.resolution = resolution
        self.tol = tol
        self.has_subclasses = False
        self.auto_combine_mms = auto_combine_mms

        # Define possibly unspecfied values
        if self.biases is None and self.weights is not None:
            self.biases = np.zeros_like(self.weights[:, [0]])
        if self.offsets is None and self.normals is not None:
            self.offsets = np.zeros_like(self.normals[:, [0]])
        if self.steepness is None:
            self.steepness = np.array([1])

        # Get normals if Poly specified
        if self.poly is not None:
            self.normals, self.offsets = normals_from_polygon(self.poly)

        # Get softmax weights
        if self.weights is not None:
            self.num_classes = self.weights.shape[0]
            self.num_params = self.weights.shape[1]
        else:
            self.num_classes = self.normals.shape[0]
            self.num_params = self.normals.shape[1]
            self._weights_from_normals()

        # Modify steepness
        # <>TODO: flesh out the general case of steepnesses not based on interior class
        self.weights = (self.steepness * self.weights.T).T
        self.biases = (self.steepness * self.biases.T).T

        # Define softmax classes
        self._define_classes(labels)

    def probability(self, state=None, class_=None, find_class_probs=True,
                    find_subclass_probs=False):
        """Map the state space to probabilities.

        If the `state` parameter is None, calculate the probability for the
        specified class over the entire state space.

        If the `class_` parameter is None, calculate the probability for all
        classes over the specified state.

        If both `state` and `class_` are None, calculate the probability for
        all classes over the entire state space

        Parameters
        ----------
        state : array_like, optional
            A a 1-dimensional array containing the state values of all `N`
            states at which to find the probabilities of each class
            specified. Defaults to None, in which case
        class : int or str, optional
            A string or integer identifying the class or subclass to check the
            probability of (at a given state or over the entire state if
            no state is specified).

        Returns
        -------
        An array_like object containing the probabilities of the class(es)
        specified.

        """

        # Define the state-space
        if state is None:  # Assume we're asking about the entire state space
            # Lazily create state space
            if not hasattr(self, 'state'):
                self._define_state(self.state_spec, self.resolution)
            state = self.state
            using_state_space = True
        else:  # Use a specific state
            state = np.array(state)
            state = state.reshape(-1,self.weights.shape[1])
            using_state_space = False

        # Define the classes to be looped over for the normalizer
        if hasattr(self,'subclasses'):
            all_classes = self.subclasses
        else:
            all_classes = self.classes

        # Define the classes to be looped over for the output
        if class_ is None:  # Use all classes/subclasses
            if find_subclass_probs:
                try:
                    subclasses = self.subclasses
                except AttributeError:
                    subclasses = self.classes
                    logging.warn('No subclasses available. Using classes '
                                 'instead.')
            if find_class_probs:
                classes = self.classes
        else:
            # Convert ID to string for lookup
            if type(class_) == int or type(class_) == float:
                class_ = self.class_labels[class_]

            # Make sure it's iterable, even if just one class
            classes = {}
            try:
                classes[class_] = self.classes[class_]
            except KeyError:
                logging.debug('Couldn\'t find class {}. Looking in subclasses.'
                              .format(class_))
                classes[class_] = self.subclasses[class_]
            except e:
                logging.error('Couldn\'t find {} as a class or subclass.'
                               .format(class_))
                raise e
        using_all_classes = len(classes) == len(all_classes)

        # Set a few initial values
        if using_state_space:
            M = np.zeros(self.X.size)
            normalizer = np.zeros(self.X.size)
            if find_class_probs:
                self.probs = np.zeros((self.X.size, self.num_classes))
            if find_subclass_probs:
                self.subclass_probs = np.zeros((self.X.size, self.num_subclasses))
        else:
            M = 0
            normalizer = 0

        # Subtract a constant from all exponent terms to prevent overflow:
        # http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression
        for _, sm_class in all_classes.iteritems():
            m = state .dot (sm_class.weights) + sm_class.bias
            M = np.maximum(m, M)

        # Define the softmax normalization term
        for _, sm_class in all_classes.iteritems():
            exp_term = np.dot(state, sm_class.weights) + sm_class.bias
            normalizer += np.exp(exp_term - M)

        # Find class probabilities
        if find_class_probs:
            for _, sm_class in classes.iteritems():

                # Find probability for superclasses or regular classes
                if len(sm_class.subclasses) > 0:
                    exp_term = 0
                    for _, subclass in sm_class.subclasses.iteritems():
                        exp_term += np.exp(np.dot(state, subclass.weights)
                                           + subclass.bias - M)
                else:
                    exp_term = np.exp(np.dot(state, sm_class.weights)
                                      + sm_class.bias - M)
                sm_class.probs = exp_term / normalizer

                # Assign probabilities to the softmax collection
                if using_state_space:
                    self.probs[:, sm_class.id] = sm_class.probs

        # Find subclass probabilities
        if find_subclass_probs:
            for _, sm_class in subclasses.iteritems():
                exp_term = np.exp(np.dot(state, sm_class.weights)
                                      + sm_class.bias - M)
                sm_class.probs = exp_term / normalizer

                # Assign probabilities to the softmax collection
                if using_state_space:
                    self.subclass_probs[:, sm_class.id] = sm_class.probs


        # Check probs to make sure everything sums to 1
        if using_state_space and using_all_classes:
            try:
                if hasattr(self,'probs'):
                    assert (np.abs(self.probs.sum(axis=1) - 1) < self.tol).all()
                if hasattr(self,'subclass_probs'):
                    assert (np.abs(self.subclass_probs.sum(axis=1) - 1)\
                            < self.tol).all()
            except AssertionError, e:
                logging.exception('Probabilites not summing to 1 at each point in '
                                  'the state!')
                raise e

        # Return the probability if looking for a single state
        if not using_state_space or not using_all_classes:
            return sm_class.probs

    def learn_from_data(self, data):
        """Learn a softmax model from a given dataset.

        """
        pass

    def add_classes(self, weights, biases, labels=None, steepness=1):
        """Add m>=1 classes to the current Softmax model.
        """
        self.weights = np.vstack((self.weights, weights))
        self.biases = np.hstack((self.biases, biases))
        self.steepness = np.hstack((self.steepness, steepness))
        self.num_classes += biases.size

        if labels is None:
            labels = []
            if hasattr(self, 'subclasses'):
                j = self.num_subclasses
            else:
                j = self.num_classes
            for i in range(biases.size):
                labels[i] = 'Class {}'.format(j + i)

        if hasattr(self, 'subclasses'):
            all_labels = self.subclass_labels + labels
        else:
            all_labels = self.class_labels + labels

        # Remove previous attributes assigned by _define_classes and combinemms
        if hasattr(self,'probs'):
            del self.probs
        if hasattr(self,'subclass_probs'):
            del self.subclass_probs

        del self.classes
        del self.class_labels
        if hasattr(self, 'subclasses'):
            del self.subclasses
            del self.subclass_labels

        self.num_classes = len(all_labels)

        self._define_classes(all_labels)

    def check_normals(self):
        """Ensures that all normals are correctly formed

        Especially that each dimension of all normals sums to zero.

        """
        # Check the augmented matrix A = [W | n] to see if W and n are
        # linearly dependent
        # Check that all n_x, n_y sum to 0

        # <>TODO: implement this function!
        # Get all normal vectors between *all* classes
        for class_weight in self.weights:
            pass

    def get_boundary(self, class1, class2=None, log_odds_ratio=0):
        # <>TODO: implement this!
        pass

    def move(self, new_pose=None, translation=None, rotation=None,
             *args, **kwargs):
        # Make sure to save original weights & biases
        if not hasattr(self, 'original_weights'):
            self.original_weights = np.copy(self.weights)
            self.original_biases = np.copy(self.biases)

        # If first 2 positional arguments specified, assume relative movement
        if not (new_pose is None) and not (translation is None):
            logging.debug('Assuming relative movement of softmax distribution.')
            rotation = translation
            translation = new_pose
            new_pose = None

        # Move relative or absolute, depending on the input
        if new_pose is None:
            self._move_relative(translation, rotation, *args, **kwargs)
        else:
            self._move_absolute(new_pose, *args, **kwargs)

    def _move_absolute(self, new_pose, rotation_unit='degrees'):

        # Validate inputs
        new_pose = np.array(new_pose).astype('float')
        if new_pose.size < 2 or new_pose.size > 3:
            raise ValueError('Pose should have 2 or 3 values. {} was given.'
                             .format(new_pose))
        elif new_pose.size == 2:
            # Assume theta is 0
            new_pose = np.hstack((new_pose, np.zeros(1)))

        # Convert theta to radians
        if rotation_unit == 'degrees':
                new_pose[2] = np.deg2rad(new_pose[2])

        # Convert scalar rotation to rotation matrix
        r = -new_pose[2]  # ccw rotation
        rotation = np.array([[np.cos(r), -np.sin(r)],
                            [np.sin(r), np.cos(r)]
                            ])

        # Create translation (inverted) 
        translation = -new_pose[0:2]

        # <>TODO: add rotation point
        # Rotate about the origin
        for i, weights in enumerate(self.original_weights):
            self.weights[i] = weights.dot(rotation)

        # Translate
        self.biases = self.original_biases + self.weights .dot (translation)

        # Redefine softmax classes
        if hasattr(self,'probs'):
            del self.probs
        if hasattr(self,'subclass_probs'):
            del self.subclass_probs
        self._combine_mms()

        # Move polygon
        if hasattr(self,'poly'):
            from cops_and_robots.map_tools.map_elements import MapObject
            mo = MapObject('', self.poly.exterior.coords[:], pose=new_pose,
                           has_relations=False)
            self.poly = mo.shape


    def _move_relative(self, translation=None, rotation=None,
                       rotation_point=None, rotation_unit='degrees',
                       reversed_translation=True, rotate_ccw=True):

        # Validate inputs
        if rotation_point is None:
            rotation_point = np.zeros(self.weights.shape[1])
        if translation is None: 
            translation = np.zeros(self.weights.shape[1])
        if rotation is None:
            rotation = np.eye(self.weights.shape[1])

        # Ensure inputs are proper numpy arrays
        translation = np.asarray(translation)
        translation.reshape(1, self.weights.shape[1])
        rotation = np.asarray(rotation)
        rotation_point = np.asarray(rotation_point)

        # Apply move_relative settings
        if reversed_translation:
            translation = -translation
        if rotate_ccw:
            rotation = -rotation

        # Convert scalar rotation to rotation matrix
        if np.array(rotation).ndim == 0:
            r = rotation
            if rotation_unit == 'degrees':
                r = np.deg2rad(r)
            rotation = np.array([[np.cos(r), -np.sin(r)],
                                [np.sin(r), np.cos(r)]
                                ])

        # Check if translation and rotation are well-formed
        try:
            assert translation.shape[0] == self.weights.shape[1]
        except AssertionError, e:
            logging.exception('Translation not well formed. Trying to '
                'translate weights \n{}\n by translation \n{}.'
                .format(self.weights, translation))
            raise e

        try:
            assert rotation.shape[0] == self.weights.shape[1]
            assert rotation.shape[1] == self.weights.shape[1]
        except AssertionError, e:
            logging.exception('Rotation not well formed. Trying to '
                'rotate weights \n{}\n by rotation \n{}.'
                .format(self.weights, translation))
            raise e

        # Rotate about the rotation point
        # Translate back to the origin
        self.biases = self.biases + self.weights .dot (rotation_point)

        # Rotate
        for i, weights in enumerate(self.weights):
            self.weights[i] = weights.dot(rotation)

        # Translate back to the original point, then apply actual translation
        self.biases = self.biases + self.weights .dot (-rotation_point)
        self.biases = self.biases + self.weights .dot (translation)

        # Redefine softmax classes
        if hasattr(self,'probs'):
            del self.probs
        if hasattr(self,'subclass_probs'):
            del self.subclass_probs
        self._combine_mms()

    def _combine_mms(self):
        """Combine classes with the same label.

        Biases, bounds, class_cmaps, class_colors, labels, state an weights
        don't change.

        classes refers to the MMS classes, whereas subclasses refers to all
        classes (even those that share a label).

        """
        from softmax_class import SoftmaxClass

        # Find unique class_labels
        unique_labels = []
        for class_label in self.classes:
            i = class_label.find('__')
            if i != -1:
                class_label = class_label[:i]
            unique_labels.append(class_label)
        unique_labels = list(set(unique_labels))

        if len(unique_labels) == len(self.classes) \
            and not hasattr(self, 'subclasses'):
            self.has_subclasses = False
            logging.debug('Doesn\'t have subclasses - no combinations needed!')
            return
        else:
            self.has_subclasses = True

        # Convert classes to subclasses (unless subclasses already exist)
        if not hasattr(self, 'subclasses'):
            self.subclasses = self.classes.copy()
            self.subclass_labels = self.class_labels
        self.num_subclasses = len(self.subclasses)

        # Re-add subclasses to class list
        self.classes = {}
        self.class_labels = []
        self.num_classes = 0

        for i, label in enumerate(self.subclass_labels):

            # Get rid of the subclass' number suffix
            full_label = label
            suffix_i = label.find('__')
            if suffix_i != -1:
                label = label[:suffix_i]
            j = self.num_classes

            # Create new class if this subclass hasn't yet been seen
            if label not in self.classes:

                 self.class_labels.append(label)
                 #<>TODO: add all weights/biases from subclasses
                 self.classes[label] = SoftmaxClass(id_=j,
                                                    label=label,
                                                    weights=self.weights[j],
                                                    bias=self.biases[j],
                                                    color=self.class_colors[j],
                                                    cmap=self.class_cmaps[j],
                                                    softmax_collection=self,
                                                    )
                 self.num_classes += 1

            # Add subclass to each class object
            subclass = self.subclasses[full_label]
            subclass.weights = self.weights[i]
            subclass.bias = self.biases[i]
            self.classes[label].add_subclass(subclass)

    def _define_state(self, state_spec, res, bounds=None):
        """Create a numeric state vector from a specification.

        Possible specifications:
        * 'x'
        * 'x y'
        * 'x x_dot'
        * 'x y_dot'
        * 'x y x_dot'
        * 'x y y_dot'
        * 'x y x_dot y_dot'
        * 'x y x^2 y^2 2xy'

        Example
        -------
        >>> self._define_state('x y')
        >>> print(self.X.shape)
        (101, 101)
        >>> print(self.Y.shape)
        (101, 101)
        >>> print(self.state.shape)
        (10201, 3)

        """

        # <>TODO: Fix for n-dimensional
        # <>TODO: Use SymPy
        # Define distribution over a gridded state space
        if bounds is None:
            bounds = self.bounds[:]
        elif self.bounds is None:
            bounds = [-5, -5, 5, 5]
            self.bounds = bounds
        else:
            self.bounds = bounds
        if state_spec == 'x':
            #<>TODO: implement bound dimension checking
            self.X = np.linspace(bounds[0], bounds[2],
                                 1 / (res ** 2))[np.newaxis]
            self.state = self.X.T
            self.ndim = 1
        elif state_spec == 'x y':
            self.X, self.Y = np.mgrid[bounds[0]:bounds[2] + res:res,
                                      bounds[1]:bounds[3] + res:res]
            self.state = np.dstack((self.X, self.Y))
            self.state = np.reshape(self.state, (self.X.size, 2))
            self.ndim = 2
        elif state_spec == 'x x_dot':
            self.X, self.X_dot = np.mgrid[bounds[0]:bounds[2] + res:res,
                                          bounds[1]:bounds[3] + res:res]
            self.state = np.dstack((self.X, self.X_dot))
            self.state = np.reshape(self.state, (self.X.size, 2))
            self.ndim = 2
        elif state_spec == 'x y_dot':
            self.X, self.Y_dot = np.mgrid[bounds[0]:bounds[2] + res:res,
                                          bounds[1]:bounds[3] + res:res]
            self.state = np.dstack((self.X, self.Y_dot))
            self.state = np.reshape(self.state, (self.X.size, 2))
            self.ndim = 2
        elif state_spec == 'x y x_dot':
            self.X, self.Y, self.X_dot = \
                np.mgrid[bounds[0]:bounds[3] + res:res,
                         bounds[1]:bounds[4] + res:res,
                         bounds[2]:bounds[5] + res:res,
                         ]
            self.state = np.dstack((self.X, self.Y, self.X_dot))
            self.state = np.reshape(self.state, (self.X.size, 3))
            self.ndim = 3
        elif state_spec == 'x y y_dot':
            self.X, self.Y, self.Y_dot = \
                np.mgrid[bounds[0]:bounds[3] + res:res,
                         bounds[1]:bounds[4] + res:res,
                         bounds[2]:bounds[5] + res:res,
                         ]
            self.state = np.dstack((self.X, self.Y, self.Y_dot))
            self.state = np.reshape(self.state, (self.X.size, 3))
            self.ndim = 3
        elif state_spec == 'x y x_dot y_dot':
            self.X, self.Y, self.X_dot, self.Y_dot = \
                np.mgrid[bounds[0]:bounds[4] + res:res,
                         bounds[1]:bounds[5] + res:res,
                         bounds[2]:bounds[6] + res:res,
                         bounds[3]:bounds[7] + res:res,
                         ]
            self.state = np.dstack((self.X, self.Y, self.X_dot, self.Y_dot))
            self.state = np.reshape(self.state, (self.X.size, 4))
            self.ndim = 4
        elif state_spec == 'x y x^2 y^2 2xy':
            self.ndim = 4
            pass

    def _weights_from_normals(self, interior=True):
        """Create a softmax distributions from a set of normals.

        Note
        ____
        This assumes this is all with respect to an arbitrary zero
        class at the origin.

        Parameters
        ----------
        interior: bool, optional
            Whether or not to generate a new interior class. Defaults to true.

        """

        #<>TODO: generalize this for non-arbitrary stuff
        # Derive weights from normals
        if interior:
            self.weights = np.vstack((np.zeros(self.num_params), self.normals))
            self.biases = np.hstack((np.zeros(1), self.offsets))
            self.num_classes = self.num_classes + 1
        else:
            self.weights = self.normals
            self.biases = self.offsets

        # logging.debug("Weights generated from normals:\n {}"
        #              .format(self.weights))
        # logging.debug("Biases generated from normals:\n {}"
        #              .format(self.biases))

    def _define_classes(self, labels=None):
        """Sets labels and colors for all classes.

        """
        from cops_and_robots.fusion.softmax import SoftmaxClass

        # Check for unique labels
        if labels is not None:
            # Define a counter for non-unique label names
            label_counts = {}
            new_labels = labels[:]

            for i, label in enumerate(labels):
                other_labels = labels[:i] + labels[i + 1:]
                if label in other_labels:
                    try:
                        label_counts[label] += 1
                    except KeyError:
                        label_counts[label] = 0
                    label = label + '__' + str(label_counts[label])
                    new_labels[i] = label
            self.class_labels = new_labels
        else:
            self.class_labels = ['Class {}'.format(i + 1)
                                 for i in range(0, self.num_classes)]

        # Make sure we have as many colors as classes
        while self.num_classes > len(Softmax.class_cmaps):
            Softmax.class_cmaps += Softmax.class_cmaps
            Softmax.class_colors += Softmax.class_colors
        self.class_cmaps = Softmax.class_cmaps[0:self.num_classes]
        self.class_colors = Softmax.class_colors[0:self.num_classes]

        # Define individual classes
        self.classes = {}
        for i, label in enumerate(self.class_labels):
            self.classes[label] = SoftmaxClass(id_=i,
                                               label=label,
                                               weights=self.weights[i],
                                               bias=self.biases[i],
                                               color=self.class_colors[i],
                                               cmap=self.class_cmaps[i],
                                               softmax_collection=self,
                                               )

        if self.auto_combine_mms:
            self._combine_mms()


def normals_from_polygon(polygon):
    """Get all unit normal vectors from the exterior of a polygon.

    Parameters
    ----------
        polygon : shapely.Polygon
            A simple polygon.

    Returns
    -------
    array_like
        List of unit vectors.

    """
    # <>TODO: Extend to 3D

    centroid = np.array(polygon.centroid)
    pts = polygon.exterior.coords[:] - centroid
    n_faces = len(pts) - 1
    normals = np.zeros((n_faces, 2))  # <>EXTEND
    offsets = np.zeros(n_faces)
    m = len(pts[0]) + 1

    for i in range(n_faces):
        midpoint = pts[i] + np.subtract(pts[i + 1], pts[i]) / 2.0
        P = np.vstack((pts[i], midpoint, pts[i + 1]))
        P = np.hstack((P, np.ones((m, 1))))

        # Use SVD to find nullspace (normals)
        u, s, v = np.linalg.svd(P)

        # <>TODO: look for degenerate cases
        tolerance = 10 ** -6
        j = int(np.argwhere(np.abs(s) < tolerance))
        sign = np.sign(v[j, -1]) * -1
        normals[i, :] = v[j, :-1] * sign
        offsets[i] = v[j, -1] * sign - normals[i, :].dot(centroid)

        if np.sum(np.dot(P, v[j, :])) > tolerance:
            logging.warning('Not a well-defined nullspace!')

    # logging.debug("Normals generated via SVD: \n{}".format(normals))
    # logging.debug("Offsets generated via SVD: {}".format(offsets))

    return normals, offsets

# # Load functions from external files
# from cops_and_robots.fusion.softmax._models import *
# from cops_and_robots.fusion.softmax._synthesis import *
# import cops_and_robots.fusion.softmax as softmax


if __name__ == '__main__':
    np.set_printoptions(precision=10, suppress=True)
    from _models import demo_models

    demo_models()
    # batch_vs_lp()
    # batch_test(visualize=True, create_combinations=True)
    # measurements = ['Inside', 'Right']
    # lp_test(measurements, verbose=False, visualize=True)

    # poly = box(1,1,2,3)
    # container_poly = box(-3,-4,3,4)
    # # container_poly = _make_regular_2D_poly(4, max_r=2, theta=np.pi/4)
    # # bism = binary_intrinsic_space_model(poly=poly)
    # bism = binary_intrinsic_space_model(poly=poly, container_poly=container_poly)
    # title = 'Binary MMS Intrinsic Space Model'
    # bism.binary_models['Front'].plot(show_plot=False, title=title)
    # plt.show()

