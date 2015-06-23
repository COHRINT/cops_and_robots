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
    resolution : float, optional
        Grid resolution over which the state space is defined. Defaults to 0.1.
    class_labels : list of str, optional
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
                 bounds=[-5, -5, 5, 5], resolution=0.1, class_labels=None,
                 auto_create_mms=True):

        if weights is not None and normals is not None and poly is not None:
            raise ValueError('One of weights, normals or poly must be '
                             'specified - choose only one!')

        # Define base object attributes
        self.weights = weights
        self.biases = biases
        self.normals = normals
        self.offsets = offsets
        self.poly = poly
        self.steepness = steepness
        self.bounds = bounds
        self.state_spec = state_spec

        # Define possibly unspecfied values
        if self.biases is None and self.weights is not None:
            self.biases = np.zeros_like(self.weights[:, [0]])
        if self.offsets is None and self.normals is not None:
            self.offsets = np.zeros_like(self.normals[:, [0]])
        if self.steepness is None:
            self.steepness = np.array([1])

        # Create state vector
        self.define_state(state_spec, resolution)

        # Create softmax distributons
        if self.poly is not None:
            self.normals, self.offsets = normals_from_polygon(self.poly)
        if self.weights is not None:
            self.num_classes = self.weights.shape[0]
            self.num_params = self.weights.shape[1]
            self.probs_from_weights()
        else:
            self.num_classes = self.normals.shape[0]
            self.num_params = self.normals.shape[1]
            self.probs_from_normals()

        # Define superclasses and all class/superclass colors
        self.set_class_labels(class_labels)

        # Combine MMS superclasses
        if self.num_classes > len(set(self.class_labels)) and auto_create_mms:
            self.combine_mms()

    def define_state(self, state_spec, res):
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
        >>> self.define_state('x y')
        >>> print(self.X.shape)
        (101, 101)
        >>> print(self.Y.shape)
        (101, 101)
        >>> print(self.state.shape)
        (10201, 3)

        """

        # <>TODO: Fix for 3D or n-dimensional
        # <>TODO: Use SymPy
        # Define distribution over a gridded state space
        bounds = self.bounds[:]
        if state_spec == 'x':
            self.X = np.linspace(bounds[0], bounds[2],
                                 1 / (res ** 2))[np.newaxis]
            self.state = self.X.T
        elif state_spec == 'x y':
            self.X, self.Y = np.mgrid[bounds[0]:bounds[2] + res:res,
                                      bounds[1]:bounds[3] + res:res]
            self.state = np.dstack((self.X, self.Y))
            self.state = np.reshape(self.state, (self.X.size, 2))
        elif state_spec == 'x x_dot':
            pass
        elif state_spec == 'x y_dot':
            pass
        elif state_spec == 'x y x_dot':
            pass
        elif state_spec == 'x y y_dot':
            pass
        elif state_spec == 'x y x_dot y_dot':
            pass
        elif state_spec == 'x y x^2 y^2 2xy':
            pass

    def probs_from_normals(self, interior=True):
        """Create a softmax distributions from a set of normals.

        Parameters
        ----------
        interior: bool, optional
            Whether or not to generate a new interior class. Defaults to true.

        """
        # Derive weights from normals
        self.weights = np.vstack((np.zeros(self.num_params), self.normals))
        self.biases = np.hstack((np.zeros(1), self.offsets))

        # Create the Softmax distribution from the weights we found
        self.num_classes = self.num_classes + 1
        self.probs_from_weights()

        logging.debug("Weights generated from normals:\n {}"
                     .format(self.weights))
        logging.debug("Biases generated from normals:\n {}"
                     .format(self.biases))

    def probs_from_weights(self):
        """Create a softmax distributions from a set of weights.

        """
        # Modify steepness
        self.weights = (self.steepness * self.weights.T).T
        self.biases = (self.steepness * self.biases.T).T

        # <>TODO:
        # Subtract a constant from all weights to prevent overflow:
        # http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression

        # Define the softmax normalization term
        sum_ = np.zeros((self.num_classes, self.X.size))
        for i, weights in enumerate(self.weights):
            exp_term = np.dot(self.state, weights) + self.biases[i]
            sum_[i, :] = np.exp(exp_term)
        normalizer = sum(sum_)  # scalar value

        # Define each class' probability
        self.probs = np.zeros((self.X.size, self.num_classes))
        for i in range(self.num_classes):
            exp_term = np.dot(self.state, self.weights[i, :]) + self.biases[i]
            self.probs[:, i] = np.exp(exp_term) / normalizer

        # Check probs to make sure everything sums to 1
        if ~(np.all(np.sum(self.probs))):
            logging.warning('Probabilites not summing to 1 at each point in'
                            'the state!')

    def probs_at_state(self, state, class_=None,):
        """Find the probabilities for each class for a given state.

        Parameters
        ----------
        state : array_like
            A a 1-dimensional array containing the state values of all `N`
            states at which to find the probabilities of each class
            specified.
        class_ : int, optional
            The zero-indexed ID of the class.

            Defaults to `None`, which will provide all classes' probabilities.

        Returns
        -------
        An array_like object containing the probabilities of the class(es)
        specified.

        """

        # <>TODO: Subtract a constant from all weights to prevent overflow:
        # http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression

        # <>TODO: Refactor for MMS class

        if type(class_) is str:
            class_ = self.class_labels.index(class_)

        if hasattr(self, 'subclass_weights'):
            weights = self.subclass_weights[:]

            # Define the softmax normalizer
            sum_ = np.zeros(weights.shape[0])
            for i, weight in enumerate(weights):
                exp_term = np.dot(state, weight) + self.biases[i]
                sum_[i] = np.exp(exp_term)
            normalizer = sum(sum_)  # scalar value

            # Define each class' probability
            probs = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                exp_term = 0

                #Use all related subclasses
                for j in range(self.num_subclasses):
                    if self.class_labels[i] == self.subclass_labels[j]:
                        exp_term += np.exp(np.dot(state, self.subclass_weights[j, :])\
                            + self.subclass_biases[j])
                probs[i] = exp_term / normalizer
        else:
            weights = self.weights[:]

            # Define the softmax normalizer
            sum_ = np.zeros(weights.shape[0])
            for i, weight in enumerate(weights):
                exp_term = np.dot(state, weight) + self.biases[i]
                sum_[i] = np.exp(exp_term)
            normalizer = sum(sum_)  # scalar value

            # Define each class' probability
            probs = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                exp_term = np.dot(state, self.weights[i, :]) + self.biases[i]
                probs[i] = np.exp(exp_term) / normalizer

        # Check probs to make sure everything sums to 1
        if ~(np.all(np.sum(probs))):
            logging.warning('Probabilites not summing to 1 at each point in'
                            'the state!')
        if class_ is None:
            return probs
        else:
            return probs[class_]

    def combine_mms(self):
        """Combine classes with the same label.

        """
        # Convert classes to subclasses
        self.subclass_probs = self.probs[:]
        self.subclass_labels = self.class_labels[:]
        self.subclass_colors = self.class_colors[:]
        self.subclass_cmaps = self.class_cmaps[:]
        self.subclass_weights = self.weights[:]
        self.subclass_biases = self.biases[:]

        self.num_subclasses = self.subclass_probs.shape[1]
        self.num_classes = len(set(self.class_labels))
        self.class_labels = []
        for l in self.subclass_labels:
            if l not in self.class_labels:
                self.class_labels.append(l)

        # Assign new colors to unique classes
        self.class_cmaps = []
        self.class_colors = []
        for i, label in enumerate(self.class_labels):
            self.class_cmaps.append(
                next(self.subclass_cmaps[j]
                    for j, slabel in enumerate(self.subclass_labels)
                    if label == slabel))
            self.class_colors.append(
                next(self.subclass_colors[j]
                    for j, slabel in enumerate(self.subclass_labels)
                    if label == slabel))

        # Merge probabilities from subclasses
        j = 0
        h = 0
        remaining_labels = self.subclass_labels[:]
        old_label = ''  # <>TODO: get rid of this. TLC would find it unpretty.
        remaining_probs = self.subclass_probs[:]
        self.probs = np.zeros((self.subclass_probs.shape[0], self.num_classes))
        for label in remaining_labels:
            indices = [k for k, other_label in enumerate(remaining_labels)
                       if label == other_label]
            probs = remaining_probs[:, indices]
            self.probs[:, h] += np.sum(probs, axis=1)

            remaining_labels = [remaining_labels[k]
                                for k, _ in enumerate(remaining_labels)
                                if k not in indices]
            remaining_probs = np.delete(remaining_probs, indices, axis=1)
            if len(remaining_labels) == 0:
                break
            else:
                j += 1
                if label != old_label:
                    h += 1
                old_label = label

    def weights_by_label(self, label):
        weights = []
        biases = []
        if hasattr(self, 'subclass_weights'):
            for i, subclass_label in enumerate(self.subclass_labels):
                if label == subclass_label:
                    weights.append(self.subclass_weights[i])
                    biases.append(self.subclass_biases[i])
        else:
            for i, class_labels in enumerate(self.class_labels):
                if label == class_labels:
                    weights.append(self.weights[i])
                    biases.append(self.biases[i])
        return weights, biases

    def add_classes(self, weights, biases, labels=None, steepness=1):
        """Add m>=1 classes to the current Softmax model.
        """
        self.weights = np.vstack((self.weights, weights))
        self.biases = np.hstack((self.biases, biases))
        self.steepness = np.hstack((self.steepness, steepness))
        self.num_classes += biases.size

        if labels is None:
            labels = ['Class {}'.format(self.num_classes)]

        new_class_labels = self.class_labels + labels
        self.set_class_labels(new_class_labels)

        self.probs_from_weights()

    def set_class_labels(self, class_labels=None):
        """Sets label and label colors for all classes and subclasses.

        """
        # Make sure we have as many colors as classes
        while self.num_classes > len(Softmax.class_cmaps):
            Softmax.class_cmaps += Softmax.class_cmaps
            Softmax.class_colors += Softmax.class_colors

        if class_labels is not None:
            self.class_labels = class_labels

            # Assign new colors to unique superclasses
            self.class_cmaps = [None] * self.num_classes
            self.class_colors = [None] * self.num_classes
            j = 0
            for i in range(0, self.num_classes):
                self.class_cmaps[i] = Softmax.class_cmaps[j]
                self.class_colors[i] = Softmax.class_colors[j]
                if i == self.num_classes - 1:
                    break
                elif self.class_labels[i] != self.class_labels[i + 1]:
                    j += 1
        else:
            self.class_labels = ['Class {}'.format(i + 1)
                                 for i in range(0, self.num_classes)]
            self.class_cmaps = Softmax.class_cmaps[0:self.num_classes]
            self.class_colors = Softmax.class_colors[0:self.num_classes]

    def plot_class(self, class_i, ax=None, plot_3d=False, fill_between=True, **kwargs):
        
        if self.state.shape[1] > 1:
            if ax is None:
                fig = plt.figure(1, figsize=(8, 8))
                if plot_3d:
                    ax = fig.gca(projection='3d')
                else:
                    ax = fig.gca()

            # Plot surface
            prob = self.probs[:, class_i].reshape(self.X.shape)
            
            if plot_3d:
                ax.plot_surface(self.X, self.Y, prob, rstride=1, cstride=3,
                                cmap=plt.get_cmap('jet'), alpha=0.8, vmin=0,
                                vmax=1.2, linewidth=0, antialiased=True,
                                shade=False, **kwargs)
            else:
                ax.contourf(self.X, self.Y, prob, cmap=plt.get_cmap('jet'),
                            **kwargs)

            ax.set_xlim(self.bounds[0], self.bounds[2])
            ax.set_ylim(self.bounds[1], self.bounds[3])

            # Shrink current axis's height by 10% on the bottom
            if plot_3d:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                 box.width, box.height * 0.9])

            ax.set_xlabel('x1')
            ax.set_ylabel('x2')

            if plot_3d:
                ax.set_zlabel('Probability of {}'
                              .format(self.class_labels[class_i]))
            ax.set_title('Class Probailities')

        else:
            if ax is None:
                fig = plt.figure(1, figsize=(8, 8))
                ax = fig.gca()

            # Plot curve
            ax.plot(self.X[0, :], self.probs[:, class_i],
                    color=self.class_colors[class_i], **kwargs)
            if fill_between:
                ax.fill_between(self.X[0, :], 0, self.probs[:, i],
                                color=self.class_colors[class_i],
                                alpha=0.4)
            
            ax.set_xlim(self.bounds[0], self.bounds[2])

            ax.set_xlabel('x')
            ax.set_ylabel('Probability of {}'
                          .format(self.class_labels[class_i]))
            # ax.set_title('Probability of {}'
            #              .format(self.class_labels[class_i]))
        
        # plt.show()
        return ax 

    def plot(self, plot_classes=True, plot_probs=True, plot_poly=False,
             plot_normals=False, title='Softmax Classification', **kwargs):
        """Display the class and/or PDF plots of the Softmax distribution.

        The class plot shows only the critical classes (those that have the
        greatest probability at any given state).

        Parameters
        ----------
        plot_classes : bool, optional
            Plot the critical classes. Defaults to `True`.
        plot_probs : bool, optional
            Plot the probability densities. Defaults to `True`.
        plot_poly : bool, optional
            Plot the polygon from which the boundaries are formed. Defaults to
            `False`.
        **kwargs
            Keyword arguments for ``plot_classes``.
        """
        # Plotting attributes

        if plot_classes and plot_probs:
            self.fig = plt.figure(3, figsize=(14, 8))
            bbox_size = (-1.3, -0.175, 2.2, -0.075)

            ax1 = self.fig.add_subplot(1, 2, 1)
            if self.state.shape[1] > 1:
                ax2 = self.fig.add_subplot(1, 2, 2, projection='3d')
            else:
                ax2 = self.fig.add_subplot(1, 2, 2)
            self._plot_classes(ax1)
            self._plot_probs(ax2)
            
        elif plot_classes:
            self.fig = plt.figure(3, figsize=(8, 8))
            bbox_size = (0, -0.175, 1, -0.075)

            self._plot_classes(**kwargs)
        elif plot_probs:
            self.fig = plt.figure(3, figsize=(8, 8))
            bbox_size = (0, -0.15, 1, -0.05)


            self._plot_probs()

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

        # Plot polygon, if possible
        if self.poly is not None and plot_poly and plot_classes:
            patch = PolygonPatch(self.poly, facecolor='none', zorder=2,
                                 linewidth=3, edgecolor='black',)
            ax1.add_patch(patch)

        # Plot normals, if possible
        # <>TODO fix crashing issue with vertical normals
        if self.normals is not None and plot_normals and plot_classes:
            t = np.arange(self.bounds[0], self.bounds[2] + 1)
            for i, normal in enumerate(self.normals):
                if abs(normal[1]) < 0.0001:
                    ax1.axvline(self.offsets[i], ls='--', lw=3, c='black')
                else:
                    slope = normal[0]
                    y = slope * t - self.offsets[i]
                    ax1.plot(t, y, ls='--', lw=3, c='black')
        plt.show()

    def learn_from_data(self, data):
        """Learn a softmax model from a given dataset.

        """
        pass

    def check_normals(self):
        """Ensures that all normals are correctly formed

        Especially that each dimension of all normals sums to zero.

        """
        # Check the augmented matrix A = [W | n] to see if W and n are
        # linearly dependent
        # Check that all n_x, n_y sum to 0

        # Get all normal vectors between *all* classes
        for class_weight in self.weights:
            pass

        # <>TODO: implement this function!

    def _plot_probs(self, ax=None):
        if self.state.shape[1] == 1:
            if ax is None:
                ax = self.fig.gca()
            self._plot_probs_1D(ax)
        elif self.state.ndim == 2:
            if ax is None:
                ax = self.fig.gca(projection='3d')
            self._plot_probs_2D(ax)
        elif self.state.ndim == 3:
            if ax is None:
                ax = self.fig.gca(projection='3d')
            self._plot_probs_3D(ax)
        else:
            raise ValueError('The state vector must be able to be represented '
                             'in 1D, 2D or 3D to be plotted.')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

    def _plot_probs_1D(self, ax):
        for i, color in enumerate(self.class_colors):
            ax.plot(self.X[0, :], self.probs[:, i], color=color)
            ax.fill_between(self.X[0, :], 0, self.probs[:, i], color=color,
                            alpha=0.4)

        ax.set_xlim(self.bounds[0], self.bounds[2])
        ax.set_ylim(0, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability P(D=i|X)')
        ax.set_title('Class Probabilities')

    def _plot_probs_2D(self, ax):
        Z = self.probs.reshape(self.X.shape[0], self.X.shape[1],
                               self.num_classes)

        plot_multisurface(self.X, self.Y, Z, ax, cstride=4, rstride=4)

        ax.set_xlim(self.bounds[0], self.bounds[2])
        ax.set_ylim(self.bounds[1], self.bounds[3])

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('Probability P(D=i|X)')
        ax.set_title('Class Probabilities')

    def _plot_probs_3D(self, ax):
        pass

    def _plot_classes(self, ax=None, plot_poly=False):
        """Plot only the critical classes.

        Critical classes are defined as the classes with highest probability
        for a given state vector `x`.

        """
        # <>TODO: Fix the checking of the state vector to allow, say, 1 x y x^2
        # Plot based on the dimension of the state vector
        if ax is None:
            ax = self.fig.gca()

        if self.state.shape[1] == 1:
            self._plot_classes_1D(ax)
        elif self.state.ndim == 2:
            self._plot_classes_2D(ax)
        elif self.state.ndim == 3:
            self._plot_classes_3D(ax)
        else:
            raise ValueError('The state vector must be able to be represented '
                             'in 1D, 2D or 3D to be plotted.')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

    def _plot_classes_1D(self, ax):

        res = 1 / self.X.size
        fake_X, fake_Y = np.mgrid[self.bounds[0]:self.bounds[2] + res:res,
                                  0:0.1]
        max_pdf_indices = np.argmax(self.probs, axis=1)
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

    def _plot_classes_2D(self, ax):
        # <>TODO: come up with more elegant solution than scatter plot
        # Identify colors of critical classes for each state
        np.set_printoptions(threshold=np.nan)
        max_pdf_indices = np.argmax(self.probs, axis=1)
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

    def _plot_classes_3D(self):
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
        logging.debug('u:\n{}'.format(u))
        logging.debug('s:\n{}'.format(s))
        logging.debug('v:\n{}'.format(v))

        tolerance = 10 ** -6
        j = int(np.argwhere(abs(s) < tolerance))
        sign = np.sign(v[j, -1]) * -1
        normals[i, :] = v[j, :-1] * sign
        offsets[i] = v[j, -1] * sign - normals[i, :].dot(centroid)

        if np.sum(np.dot(P, v[j, :])) > tolerance:
            logging.warning('Not a well-defined nullspace!')

    logging.debug("Normals generated via SVD: \n{}".format(normals))
    logging.debug("Offsets generated via SVD: {}".format(offsets))

    return normals, offsets


def make_regular_2D_poly(n_sides=3, origin=(0, 0), theta=0, max_r=1):
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
    `no-detection` subclasses, and one `detection` sublcass. This uses the
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

    camera = Softmax(poly=camera_poly, class_labels=labels,
                     steepness=steepness)
    return camera


def speed_model():
    """Generate a one-dimensional Softmax model for speeds.
    """
    labels = ['Stopped', 'Slow', 'Medium', 'Fast']
    sm = Softmax(weights=np.array([[0], [150], [175], [200]]),
                 biases=np.array([0, -2.5, -6, -14]),
                 state_spec='x', class_labels=labels,
                 bounds=[0, 0, 0.4, 0.4])
    return sm


def wall_model(l=1.2192, w=0.1524, origin=[0,0]):
    """Generate a two-dimensional Softmax model around a wall.
    """
    wall_poly = box(-w/2 + origin[0],
                    -l/2 + origin[1],
                    w/2 + origin[0],
                    l/2 + origin[1])

    steepness = [0, 10, 10, 10, 10]
    labels = ['Interior','Front', 'Left', 'Back', 'Right']
    wall = Softmax(poly=wall_poly, steepness=steepness, class_labels=labels)
    return wall

def pentagon_model():
    poly = make_regular_2D_poly(5, max_r=2, theta=np.pi/3.1)
    labels = ['Interior',
              'Mall Terrace Entrance',
              'Heliport Facade',
              'South Parking Entrance', 
              'Concourse Entrance',
              'River Terrace Entrance', 
             ]
    steepness = 5
    sm = Softmax(poly=poly, class_labels=labels, resolution=0.1, steepness=5)
    return sm

def distance_space_model(poly=None):
    if poly == None:
        poly = make_regular_2D_poly(4, max_r=2, theta=np.pi/4)
    labels = ['Inside'] + ['Near'] * 4
    steepnesses = [3] * 5
    sm = Softmax(poly=poly, class_labels=labels, resolution=0.1,
                 steepness=steepnesses, auto_create_mms=False)

    steepnesses = [3.1] * 5
    far_bounds = make_regular_2D_poly(4, max_r=3, theta=np.pi/4)
    labels = ['Inside'] + ['Outside'] * 4
    sm_far = Softmax(poly=poly, class_labels=labels, resolution=0.1,
                 steepness=steepnesses)

    new_weights = sm_far.weights[1:]
    new_biases = sm_far.biases[1:] - 0.05  # <>TODO: Fix this hack
    new_class_labels = ['Outside'] * 4
    new_steepnesses = steepnesses[1:]

    sm.add_classes(new_weights, new_biases, new_class_labels, new_steepnesses)
    sm.combine_mms()
    return sm

def intrinsic_space_model(poly=None):
    if poly == None:
        poly = make_regular_2D_poly(4, max_r=2, theta=np.pi/4)

    # <>TODO: If sides != 4, find a way to make it work!

    labels = ['Inside', 'Front', 'Left', 'Back', 'Right']
    steepness = 3
    sm = Softmax(poly=poly, class_labels=labels, resolution=0.1,
                 steepness=steepness)
    return sm

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=10, suppress=True)

    # sm = pentagon_model()
    # sm.plot()

    # sm = speed_model()
    # sm.plot()

    sm = intrinsic_space_model()
    sm.plot_class(1)
    print sm.weights
    print sm.biases

    # x = [-3, -3, 3, 3]
    # y = [-1, 1, 1, -1]
    # y = [-3, -1, -1, -3]
    # pts = zip(x,y)
    # poly = Polygon(pts)
    # sm = distance_space_model(poly)
    # sm.plot(plot_poly=True)

    # # Make big poly
    # poly = make_regular_2D_poly(n_sides=4, origin=(0,0), theta=np.pi/4, max_r=3)
    # steepness = np.array([0, 3, 3, 3, 3])
    # sm_big = Softmax(poly=poly , steepness=steepness)
    
    # # Make small poly
    # poly = make_regular_2D_poly(n_sides=4, origin=(0,0), theta=np.pi/4, max_r=1)
    # steepness = np.array([0, 3, 3, 3, 3])
    # sm_small = Softmax(poly=poly, steepness=steepness)
    
    # new_weights = sm_small.weights[1:]
    # new_biases = sm_small.biases[1:]
    # new_class_labels = ['Interior'] * 4
    # new_steepness = steepness[1:]

    # sm_big.add_classes(new_weights, new_biases, new_class_labels, new_steepness)
    # sm_big.plot(plot_poly=True)