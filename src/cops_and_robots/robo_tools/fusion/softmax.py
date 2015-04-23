#!/usr/bin/env python
"""Provides creation, maniputlation and plotting of SoftMax distributions.

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


class SoftMax(object):
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
        A string specified as defined in :func:`~softmax.SoftMax.define_state`.
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
    class_cmaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys',
                   'RdPu']
    class_colors = ['red', 'blue', 'green', 'orange', 'purple', 'grey', 'pink']

    def __init__(self, weights=None, biases=None, normals=None, offsets=None,
                 poly=None, steepness=None, rotation=None, state_spec='x y',
                 bounds=[-5, -5, 5, 5], resolution=0.1, class_labels=None):

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
        if self.num_classes > len(set(self.class_labels)):
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

        # Create the SoftMax distribution from the weights we found
        self.num_classes = self.num_classes + 1
        self.probs_from_weights()

        logging.info("Weights generated from normals:\n {}"
                     .format(self.weights))
        logging.info("Biases generated from normals:\n {}"
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

    def probs_at_state(self, state, class_=None):
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

        # <>TODO:
        # Subtract a constant from all weights to prevent overflow:
        # http://ufldl.stanford.edu/wiki/index.php/Exercise:Softmax_Regression

        # <>TODO:
        # Allow for class labels to be used instead of class IDs.
        
        # <>TODO:
        # Fix for MMS models

        # Define the softmax normalizer
        sum_ = np.zeros(self.weights.shape[0])
        for i, weights in enumerate(self.weights):
            exp_term = np.dot(state, weights) + self.biases[i]
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

        self.num_subclasses = self.subclass_probs.shape[1]
        self.num_classes = len(set(self.class_labels))

        # Assign new colors to unique classes
        self.class_cmaps = self.class_cmaps[:self.num_classes]
        self.class_colors = self.class_colors[:self.num_classes]

        # Merge probabilities from subclasses
        j = 0
        remaining_labels = self.subclass_labels[:]
        remaining_probs = self.subclass_probs[:]
        self.probs = np.zeros((self.subclass_probs.shape[0], self.num_classes))
        for label in remaining_labels:
            indices = [k for k, other_label in enumerate(remaining_labels)
                       if label == other_label]
            probs = remaining_probs[:, indices]
            self.probs[:, j] = np.sum(probs, axis=1)

            remaining_labels = [remaining_labels[k]
                                for k, _ in enumerate(remaining_labels)
                                if k not in indices]
            remaining_probs = np.delete(remaining_probs, indices, axis=1)
            if len(remaining_labels) == 0:
                break
            else:
                j += 1

    def set_class_labels(self, class_labels=None):
        """Sets label and label colors for all classes and subclasses.

        """
        # Make sure we have as many colors as classes
        while self.num_classes > len(SoftMax.class_cmaps):
            SoftMax.class_cmaps += SoftMax.class_cmaps
            SoftMax.class_colors += SoftMax.class_colors

        if class_labels is not None:
            self.class_labels = class_labels

            # Assign new colors to unique superclasses
            self.class_cmaps = [None] * self.num_classes
            self.class_colors = [None] * self.num_classes
            j = 0
            for i in range(0, self.num_classes):
                self.class_cmaps[i] = SoftMax.class_cmaps[j]
                self.class_colors[i] = SoftMax.class_colors[j]
                if i == self.num_classes - 1:
                    break
                elif self.class_labels[i] != self.class_labels[i + 1]:
                    j += 1
        else:
            self.class_labels = ['Class {}'.format(i + 1)
                                 for i in range(0, self.num_classes)]
            self.class_cmaps = SoftMax.class_cmaps[0:self.num_classes]
            self.class_colors = SoftMax.class_colors[0:self.num_classes]

    def plot_class(self, class_i):
        fig = plt.figure(1, figsize=(12, 10))
        ax = fig.gca(projection='3d')

        # Plot each surface
        prob = self.probs[:, class_i].reshape(self.X.shape)
        ax.plot_surface(self.X, self.Y, prob, rstride=1, cstride=3,
                        cmap=plt.get_cmap('jet'), alpha=0.8, vmin=0,
                        vmax=1.2, linewidth=0, antialiased=True, shade=False)

        ax.set_xlim(self.bounds[0], self.bounds[2])
        ax.set_ylim(self.bounds[1], self.bounds[3])

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('Probability of {}'.format(self.class_labels[class_i]))
        ax.set_title('Class Probailities')

        plt.show()

    def plot(self, plot_classes=True, plot_probs=True, plot_poly=False,
             plot_normals=False, title='SoftMax Classification', **kwargs):
        """Display the class and/or PDF plots of the SoftMax distribution.

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
        self.fig = plt.figure(3, figsize=(14, 8))

        if plot_classes and plot_probs:
            ax1 = self.fig.add_subplot(1, 2, 1)
            if self.state.shape[1] > 1:
                ax2 = self.fig.add_subplot(1, 2, 2, projection='3d')
            else:
                ax2 = self.fig.add_subplot(1, 2, 2)
            self._plot_classes(ax1)
            self._plot_probs(ax2)
        elif plot_classes:
            self._plot_classes(**kwargs)
        elif plot_probs:
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
                   bbox_to_anchor=(-1.1, -0.175, 2, -0.075), borderaxespad=0.)
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
        steepness=[100, 100, 2.5, 100, 100]
        labels = ['Detection', 'No Detection', 'No Detection',  'No Detection', 
              'No Detection']
    else:
        steepness=[100, 100, 2.5, 100,]
        labels = ['Detection', 'No Detection',  'No Detection', 'No Detection']        
    
    camera = SoftMax(poly=camera_poly, class_labels=labels, 
                     steepness=steepness)
    return camera


def speed_model():
    labels = ['Stopped', 'Slow', 'Medium', 'Fast']
    sm = SoftMax(weights=np.array([[0], [150], [175], [200]]),
                 biases=np.array([0, -2.5, -6, -14]),
                 state_spec='x', class_labels=labels,
                 bounds=[0, 0, 0.4, 0.4])
    return sm

def wall_model(l=1.2192, w=0.1524, origin=[0,0]):
    """Generate a two-dimensional SoftMax model around a wall.
    """
    wall = box(-w/2 + origin[0],
               -l/2 + origin[1],
                w/2 + origin[0],
                l/2 + origin[1])

    steepness = [0, 10, 10, 10, 10]
    labels = ['Interior','Front', 'Left', 'Back', 'Right']
    wall_sm = SoftMax(poly=wall, steepness=steepness, class_labels=labels)
    return wall_sm

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=2, suppress=True)

    camera = camera_model_2D()
    print(camera.probs_at_state(np.array([1, 0]), 0))
