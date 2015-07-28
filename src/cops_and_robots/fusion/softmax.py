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

from shapely.geometry import box, Polygon
from shapely.affinity import scale

import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter
from mpl_toolkits.mplot3d import Axes3D  # Seemingly unused, but for 3D plots
from descartes.patch import PolygonPatch

from cops_and_robots.helpers.visualizations import plot_multisurface

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

    def __init__(self, weights=None, biases=None, normals=None, offsets=None,
                 poly=None, steepness=None, rotation=None, state_spec='x y',
                 bounds=[-5, -5, 5, 5], resolution=0.1, labels=None,
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

    def plot(self, class_=None, show_plot=True, plot_3D=True, plot_probs=True,
             plot_dominant_classes=True, plot_poly=False, plot_normals=False,
             plot_subclasses=False, title='Softmax Classification', **kwargs):
        """Display the class and/or PDF plots of the Softmax distribution.

        The class plot shows only the critical classes (those that have the
        greatest probability at any given state).

        Parameters
        ----------
        plot_dominant_classes : bool, optional
            Plot the critical classes. Defaults to `True`.
        plot_probs : bool, optional
            Plot the probability densities. Defaults to `True`.
        plot_poly : bool, optional
            Plot the polygon from which the boundaries are formed. Defaults to
            `False`.
        **kwargs
            Keyword arguments for ``plot_dominant_classes``.
        """

        # Define probabilities lazily
        if not hasattr(self, 'probs') and not plot_subclasses:
                self.probability()
        if not hasattr(self, 'subclass_probs') and plot_subclasses:
                self.probability(find_subclass_probs=True)

        # Plotting attributes
        self.plot_3D = plot_3D
        self.plot_subclasses = plot_subclasses

        if plot_dominant_classes and plot_probs and class_ is None:
            self.fig = plt.figure(figsize=(14, 8))
            bbox_size = (-1.3, -0.175, 2.2, -0.075)

            ax1 = self.fig.add_subplot(1, 2, 1)
            if plot_3D and self.state.shape[1] > 1:
                ax2 = self.fig.add_subplot(1, 2, 2, projection='3d')
            else:
                ax2 = self.fig.add_subplot(1, 2, 2)
            self._plot_dominant_classes(ax1)
            self._plot_probs(ax2)
            axes = [ax1, ax2]

        elif plot_dominant_classes and class_ is None:
            self.fig = plt.figure(figsize=(8, 8))
            ax1 = self.fig.add_subplot(111)
            bbox_size = (0, -0.175, 1, -0.075)
            self._plot_dominant_classes(**kwargs)
            axes = [ax1]

        elif plot_probs:
            self.fig = plt.figure(figsize=(8, 8))

            if class_ is not None:
                self.classes[class_].plot(**kwargs)
                axes = [self.fig.gca()]
            else:
                if plot_3D and self.state.shape[1] > 1:
                    ax1 = self.fig.add_subplot(111, projection='3d')
                else:
                    ax1 = self.fig.add_subplot(111)
                self._plot_probs(ax1,**kwargs)
                axes = [ax1]
            bbox_size = (0, -0.15, 1, -0.05)

        # Create Proxy artists for legend labels
        proxy = [None] * self.num_classes
        for i in range(self.num_classes):
            if self.class_labels[i] not in self.class_labels[:i]:
                proxy_label = self.class_labels[i]
            else:
                proxy_label = "_nolegend_"
            proxy[i] = plt.Rectangle((0, 0), 1, 1, fc=self.class_colors[i],
                                     alpha=0.6, label=proxy_label,)

        plt.legend(handles=proxy, loc='lower center', mode='expand', ncol=4,
                   bbox_to_anchor=bbox_size, borderaxespad=0.)
        plt.suptitle(title, fontsize=16)

        # Plot polygon
        if self.poly is not None and plot_poly and plot_dominant_classes:
            patch = PolygonPatch(self.poly, facecolor='none', zorder=2,
                                 linewidth=3, edgecolor='black',)
            ax1.add_patch(patch)

        # Plot normals
        # <>TODO fix crashing issue with vertical normals
        if self.normals is not None and plot_normals and plot_dominant_classes:
            t = np.arange(self.bounds[0], self.bounds[2] + 1)
            for i, normal in enumerate(self.normals):
                if abs(normal[1]) < 0.0001:
                    ax1.axvline(self.offsets[i], ls='--', lw=3, c='black')
                else:
                    slope = normal[0]
                    y = slope * t - self.offsets[i]
                    ax1.plot(t, y, ls='--', lw=3, c='black')

        if show_plot:
            plt.show()
        return axes

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

    def _plot_probs(self, ax=None, class_=None):
        if self.state.shape[1] == 1:
            if ax is None:
                ax = self.fig.gca()
            self._plot_probs_1D(ax, class_)
        elif self.state.ndim == 2:
            if ax is None and self.plot_3D:
                ax = self.fig.gca(projection='3d')
            elif ax is None:
                ax = self.fig.gca()
            self._plot_probs_2D(ax, class_)
        elif self.state.ndim == 3:
            if ax is None:
                ax = self.fig.gca(projection='3d')
            self._plot_probs_3D(ax, class_)
        else:
            raise ValueError('The state vector must be able to be represented '
                             'in 1D, 2D or 3D to be plotted.')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

    def _plot_probs_1D(self, ax, class_):
        if type(class_) is str:
            try:
                class_ = self.classes[class_].id
            except KeyError:
                logging.debug('Couldn\'t find class {}. Looking in subclasses.'
                              .format(class_))
                class_ = self.subclasses[class_].id
                if not hasattr(self, 'subclass_probs'):
                    self.probability(find_subclass_probs=True)
                self.plot_subclasses = True
            except e:
                logging.error('Couldn\'t find {} as a class or subclass.'
                               .format(class_))
                raise e
        if self.plot_subclasses:
            Z = self.subclass_probs[:]
        else:
            Z = self.probs[:]


        if class_ is not None:
            ax.plot(self.X[0, :], Z[:,class_], color=self.class_colors[class_])
            ax.fill_between(self.X[0, :], 0, Z[:,class_], color=self.class_colors[class_],
                            alpha=0.4)
        else:
            for i in range(self.num_classes):
                ax.plot(self.X[0, :], Z[:,i], color=self.class_colors[i])
                ax.fill_between(self.X[0, :], 0, Z[:,i], color=self.class_colors[i],
                                alpha=0.4)

        ax.set_xlim(self.bounds[0], self.bounds[2])
        ax.set_ylim(0, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability P(D=i|X)')
        ax.set_title('Class Probabilities')

    def _plot_probs_2D(self, ax, class_):
        if class_ is not None:
            if type(class_) is str:
                try:
                    class_ = self.classes[class_].id
                except KeyError:
                    logging.debug('Couldn\'t find class {}. Looking in subclasses.'
                                  .format(class_))
                    class_ = self.subclasses[class_].id
                    if not hasattr(self, 'subclass_probs'):
                        self.probability(find_subclass_probs=True)
                    self.plot_subclasses = True
                except e:
                    logging.error('Couldn\'t find {} as a class or subclass.'
                                   .format(class_))
                    raise e
            if self.plot_subclasses:
                Z = self.subclass_probs[:, class_].reshape(self.X.shape[0], self.X.shape[1])
            else:
                Z = self.probs[:, class_].reshape(self.X.shape[0], self.X.shape[1])
        elif self.plot_subclasses:
            Z = self.subclass_probs.reshape(self.X.shape[0], self.X.shape[1],
                                            self.num_subclasses)
        else:
            Z = self.probs.reshape(self.X.shape[0], self.X.shape[1],
                                   self.num_classes)

        if self.plot_3D:

            if class_ is not None:
                ax.plot_surface(self.X, self.Y, Z, cstride=2, rstride=2,
                                linewidth=0, antialiased=False,
                                cmap=plt.get_cmap(self.class_cmaps[class_]))
            else:
                plot_multisurface(self.X, self.Y, Z, ax, cstride=4, rstride=4)

            ax.set_xlim(self.bounds[0], self.bounds[2])
            ax.set_ylim(self.bounds[1], self.bounds[3])
            ax.set_zlabel('Probability P(D=i|X)')
        else:
            levels = np.linspace(0, np.max(Z), 50)
            # <>TODO: Fix contourf plotting for multiple classes
            for i in range(self.num_classes):
                ax.contourf(self.X, self.Y, Z[:,:,i], levels=levels,
                            cmap=plt.get_cmap(self.class_cmaps[i]), alpha=0.8)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Class Probabilities')

    def _plot_probs_3D(self, ax, class_):
        pass

    def _plot_dominant_classes(self, ax=None, plot_poly=False):
        """Plot only the critical classes.

        Critical classes are defined as the classes with highest probability
        for a given state vector `x`.

        """
        # <>TODO: Fix the checking of the state vector to allow, say, 1 x y x^2
        # Plot based on the dimension of the state vector
        if ax is None:
            ax = self.fig.gca()

        if self.state.shape[1] == 1:
            self._plot_dominant_classes_1D(ax)
        elif self.state.ndim == 2:
            self._plot_dominant_classes_2D(ax)
        elif self.state.ndim == 3:
            self._plot_dominant_classes_3D(ax)
        else:
            raise ValueError('The state vector must be able to be represented '
                             'in 1D, 2D or 3D to be plotted.')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

    def _plot_dominant_classes_1D(self, ax):

        if self.plot_subclasses:
            probs = self.subclass_probs
        else:
            probs = self.probs

        res = 1 / self.X.size
        fake_X, fake_Y = np.mgrid[self.bounds[0]:self.bounds[2] + res:res,
                                  0:0.1]
        max_pdf_indices = np.argmax(probs, axis=1)
        max_colors = np.take(self.class_colors, max_pdf_indices)
        cc = ColorConverter()
        max_colors_rgb = np.array([cc.to_rgb(_) for _ in max_colors])

        ax.bar(self.X.T, np.ones_like(self.X.T), color=max_colors_rgb,
               linewidth=0, alpha=1)

        ax.set_xlim(self.bounds[0], self.bounds[2])
        ax.set_ylim(0, 0.01)
        ax.set_yticks([])

        ax.set_xlabel('x')
        ax.set_title('Critical Classes')

    def _plot_dominant_classes_2D(self, ax):

        if self.plot_subclasses:
            probs = self.subclass_probs
        else:
            probs = self.probs

        # <>TODO: come up with more elegant solution than scatter plot
        # Identify colors of critical classes for each state
        np.set_printoptions(threshold=np.nan)
        max_pdf_indices = np.argmax(probs, axis=1)
        max_colors = np.take(self.class_colors, max_pdf_indices)
        cc = ColorConverter()
        max_colors_rgb = np.array([cc.to_rgb(_) for _ in max_colors])

        ax.scatter(self.X, self.Y, c=max_colors_rgb, marker='s', s=100,
                   linewidths=0, alpha=1)

        ax.set_xlim(self.bounds[0], self.bounds[2])
        ax.set_ylim(self.bounds[1], self.bounds[3])
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Critical Classes')

    def _plot_dominant_classes_3D(self):
        pass


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


###############################################################################
# Individual Softmax Class
###############################################################################

class SoftmaxClass(object):
    """short description of SoftmaxClass

    long description of SoftmaxClass
    
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

    def __init__(self, id_, label, weights, bias, softmax_collection,
                 steepness=1,color='grey', cmap='Greys'):
        self.id = id_
        self.label = label
        self.weights = np.asarray(weights)
        self.bias = np.asarray(bias)
        self.softmax_collection = softmax_collection
        self.steepness = np.asarray(steepness)
        self.color = color
        self.cmap = cmap
        self.ndim = self.weights.shape[0]

        self.subclasses = {}
        self.num_subclasses = 0

    def add_subclass(self, subclass):
        """Add a subclass to this softmax class.
        """
        subclass_label = self.label
        if len(self.subclasses) > 0:

            # Number unnumbered subclass
            for sc_label, sc in self.subclasses.iteritems():
                i = sc_label.find('__')
                if i == -1:
                    old_sc = self.subclasses[sc_label]
                    del self.subclasses[sc_label]
                    sc_label = sc_label + '__0'
                    self.subclasses[sc_label] = sc

            # Number current subclass
            subclass_label = subclass_label + '__' + str(self.num_subclasses)

        self.subclasses[subclass_label] = subclass

        self.num_subclasses += 1

    def probability(self, state=None, find_subclass_probs=False):
        """Find the probability for this class at a given state.

        Parameters
        ----------
        state : array_like
            A a 1-dimensional array containing the state values of all `N`
            states at which to find the probabilities of each class
            specified.

        Returns
        -------
        An array_like object containing the probabilities of the class(es)
        specified.

        """
        p = self.softmax_collection.probability(state=state, class_=self.label,
                                                find_subclass_probs=find_subclass_probs)
        return p

    def plot(self, ax=None, fill_between=True, plot_3D=True, **kwargs):
        if self.ndim < 2:
            self._plot_1D(ax=ax, fill_between=fill_between)
        elif self.ndim == 2:
            self._plot_2D(ax=ax, plot_3D=plot_3D)
        else:
            logging.error('No plot available at higher dimensions than 2!')

    def _plot_1D(self, ax=None, fill_between=True, **kwargs):
        fig = plt.gcf()
        if ax is None:
            ax = fig.add_subplot(111)

        if hasattr(self,'probs'):
            del self.probs
        if not hasattr(self.softmax_collection, 'probs'):
            self.softmax_collection.probability()
        self.probs = self.softmax_collection.probs[:, self.id]
        self.X = self.softmax_collection.X[0,:]

        ax.plot(self.X, self.probs, color=self.color,
                **kwargs)
        if fill_between:
            ax.fill_between(self.X, 0, self.probs,
                            color=self.color, alpha=0.4)

        bounds = self.softmax_collection.bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(0, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability P(D=i|X)')
        ax.set_title('Class Probabilities')

    def _plot_2D(self, ax=None, plot_3D=True, **kwargs):
        fig = plt.gcf()
        if ax is None:
            if plot_3D:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)

        if hasattr(self,'probs'):
            del self.probs
        if not hasattr(self.softmax_collection, 'probs'):
            self.softmax_collection.probability()

        X = self.softmax_collection.X
        Y = self.softmax_collection.Y
        Z = self.softmax_collection.probs[:, self.id].reshape(X.shape[0], X.shape[1])
        bounds = self.softmax_collection.bounds

        if plot_3D:
            ax.plot_surface(X, Y, Z, cstride=2, rstride=2, linewidth=0,
                            antialiased=False, cmap=plt.get_cmap(self.cmap))

            ax.set_zlabel('Probability P(D=i|X)')
        else:
            levels = np.linspace(0, np.max(Z), 50)
            ax.contourf(X, Y, Z, levels=levels, cmap=plt.get_cmap(self.cmap),
                        alpha=0.8)

        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Class Probabilities')


###############################################################################
# Binary Softmax
###############################################################################
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

    def __init__(self, softmax_model, bounds=None):
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


###############################################################################
# Pre-defined Softmax, Multimodal Softmax and Binary Softmax models
###############################################################################

def _make_regular_2D_poly(n_sides=3, origin=(0, 0), theta=0, max_r=1):
    x = [max_r * np.cos(2 * np.pi * n / n_sides + theta) + origin[0]
         for n in range(n_sides)]
    y = [max_r * np.sin(2 * np.pi * n / n_sides + theta) + origin[1]
         for n in range(n_sides)]
    pts = zip(x,y)

    logging.debug('Generated points:\n{}'.format('\n'.join(map(str,pts))))

    return Polygon(pts)


def camera_model_2D(min_view_dist=0., max_view_dist=2):
    """Generate a softmax model for a two dimensional camera.

    A softmax model decomposing the space into five subclasses: four
    `no-detection` subclasses, and one `detection` subclass. This uses the
    Microsoft Kinect viewcone specifications of 57 degrees viewing angle.

    Classes are numbered as such:
    <>TODO: insert image of what it looks like

    Parameters
    ----------
    min_view_dist : float, optional
        The minimum view distance for the camera. Note that this is not a hard
        boundary, and specifies the distance at which an object has a 5% chance
        of detection. The default it 0 meters, which implies no minimum view
        distance.
    min_view_dist : float, optional
        The minimum view distance for the camera. Note that this is not a hard
        boundary, and specifies the distance at which an object has a 5% chance
        of detection. The default is 1 meter.

    """
    # Define view cone centered at origin
    horizontal_view_angle = 57  # from Kinect specifications (degrees)
    min_y = min_view_dist * np.tan(np.radians(horizontal_view_angle / 2))
    max_y = max_view_dist * np.tan(np.radians(horizontal_view_angle / 2))
    camera_poly = Polygon([(min_view_dist,  min_y),
                           (max_view_dist,  max_y),
                           (max_view_dist, -max_y),
                           (min_view_dist, -min_y)])

    if min_view_dist > 0:
        steepness = [100, 100, 2.5, 100, 100]
        labels = ['Detection', 'No Detection', 'No Detection', 'No Detection',
                  'No Detection']
    else:
        steepness = [100, 100, 2.5, 100,]
        labels = ['Detection', 'No Detection', 'No Detection', 'No Detection']

    bounds = [-11.25, -3.75, 3.75, 3.75]
    camera = Softmax(poly=camera_poly, labels=labels,
                     steepness=steepness, bounds=bounds)
    return camera


def speed_model():
    """Generate a one-dimensional Softmax model for speeds.
    """
    labels = ['Stopped', 'Slow', 'Medium', 'Fast']
    sm = Softmax(weights=np.array([[0], [150], [175], [200]]),
                 biases=np.array([0, -2.5, -6, -14]),
                 state_spec='x', labels=labels,
                 bounds=[0, 0, 0.4, 0.4])
    return sm


def binary_speed_model():
    sm = speed_model()
    bsm = BinarySoftmax(sm)
    return bsm


def pentagon_model():
    poly = _make_regular_2D_poly(5, max_r=2, theta=np.pi/3.1)
    labels = ['Interior',
              'Mall Terrace Entrance',
              'Heliport Facade',
              'South Parking Entrance', 
              'Concourse Entrance',
              'River Terrace Entrance', 
             ]
    steepness = 5
    sm = Softmax(poly=poly, labels=labels, resolution=0.1, steepness=5)
    return sm


def range_model(poly=None, spread=1, bounds=None):
    if poly == None:
        poly = _make_regular_2D_poly(4, max_r=2, theta=np.pi/4)
    labels = ['Inside'] + ['Near'] * 4
    steepnesses = [10] * 5
    sm = Softmax(poly=poly, labels=labels, resolution=0.1,
                 steepness=steepnesses, bounds=bounds)

    steepnesses = [11] * 5
    # far_bounds = _make_regular_2D_poly(4, max_r=3, theta=np.pi/4)
    larger_poly = scale(poly,2,2)
    labels = ['Inside'] + ['Outside'] * 4
    sm_far = Softmax(poly=poly, labels=labels, resolution=0.1,
                 steepness=steepnesses, bounds=bounds)

    new_weights = sm_far.weights[1:]
    new_biases = sm_far.biases[1:] - spread
    new_labels = ['Outside'] * 4
    new_steepnesses = steepnesses[1:]

    sm.add_classes(new_weights, new_biases, new_labels, new_steepnesses)
    return sm


def binary_range_model(poly=None, bounds=None):
    dsm = range_model(poly, bounds=bounds)
    bdsm = BinarySoftmax(dsm, bounds=bounds)
    return bdsm


def intrinsic_space_model(poly=None, bounds=None):
    if poly == None:
        poly = _make_regular_2D_poly(4, max_r=2, theta=np.pi/4)

    # <>TODO: If sides != 4, find a way to make it work!

    labels = ['Inside', 'Front', 'Left', 'Back', 'Right']
    steepness = 3
    sm = Softmax(poly=poly, labels=labels, resolution=0.1,
                 steepness=steepness, bounds=bounds)
    return sm


def binary_intrinsic_space_model(poly=None, bounds=None):
    ism = intrinsic_space_model(poly, bounds=bounds)
    bism = BinarySoftmax(ism, bounds=bounds)
    # del bism.binary_models['Inside']
    return bism


def run_demos():
    logging.info('Preparing Softmax models for demo...')
    # Regular Softmax models #################################################

    # Speed model
    sm = speed_model()
    title='Softmax Speed Model'
    logging.info('Building {}'.format(title))
    sm.plot(show_plot=False, plot_3D=False, title=title)

    # Pentagon Model
    pm = pentagon_model()
    title='Softmax Area Model (The Pentagon)'
    logging.info('Building {}'.format(title))
    pm.plot(show_plot=False, title=title)

    # Range model
    x = [-2, 0, 2, 2]
    y = [-3, -1, -1, -3]
    pts = zip(x,y)
    poly = Polygon(pts)
    rm = range_model(poly)
    rm = intrinsic_space_model(poly)
    title='Softmax Intrinsic Space Model (Irregular)'
    logging.info('Building {}'.format(title))
    rm.plot(show_plot=False, title=title)

    # Multimodal softmax models ##############################################

    # Range model (irregular poly)
    x = [-2, 0, 2, 2]
    y = [-3, -1, -1, -3]
    pts = zip(x,y)
    poly = Polygon(pts)
    rm = range_model(poly)
    title='MMS Range Model (Irregular)'
    logging.info('Building {}'.format(title))
    rm.plot(show_plot=False, title=title)

    # Camera model (with movement)
    cm = camera_model_2D()
    cm.move([-2,-2, 90])
    title='MMS Camera Model'
    logging.info('Building {}'.format(title))
    cm.plot(show_plot=False, title=title)
    cm.move([-5,-2, 90])
    title='MMS Camera Model (moved, detect only)'
    logging.info('Building {}'.format(title))
    cm.plot(show_plot=False, class_='Detection', title=title)

    # Binary softmax models ##################################################

    # Binary speed model 
    bsm = binary_speed_model()
    title='Binary MMS Speed Model'
    logging.info('Building {}'.format(title))
    bsm.binary_models['Medium'].plot(show_plot=False, title=title)

    # Binary Pentagon Model - Individual and multiple plots
    pent = pentagon_model()
    bpent = BinarySoftmax(pent)
    title='Binary MMS Pentagon Model (Not Heliport only)'
    logging.info('Building {}'.format(title))
    bpent.binary_models['Heliport Facade'].plot(show_plot=False, class_='Not Heliport Facade',
                                                title=title)
    title='Binary MMS Pentagon Model (Interior)'
    bpent.binary_models['Interior'].plot(show_plot=False, title=title)

    # Binary Range Model
    bdsm = binary_range_model()
    title='Binary MMS Range Model'
    logging.info('Building {}'.format(title))
    bdsm.binary_models['Near'].plot(show_plot=False, title=title)
    plt.show()

    # Binary intrinsic space model
    bism = binary_intrinsic_space_model()
    title='Binary MMS Intrinsic Space Model'
    logging.info('Building {}'.format(title))
    bism.binary_models['Front'].plot(show_plot=False, title=title)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=10, suppress=True)

    # run_demos()
    bism = binary_intrinsic_space_model()
    print bism.probability(class_='Front')