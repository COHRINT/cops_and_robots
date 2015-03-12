#!/usr/bin/env python
"""Generic definition of a softmax probability distribution

Softmax
=======
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
import math
import numpy as np

from shapely.geometry import box, Polygon, LineString, Point

import matplotlib.pyplot as plt
from matplotlib.colors import ColorConverter, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # Seemingly unused, but for 3D plots
from descartes.patch import PolygonPatch

import warnings  # To suppress nolabel warnings
warnings.filterwarnings("ignore", message=".*cannot be automatically added.*")


class SoftMax(object):
    """Generate a softmax distribution from weights or hyperplane normals.

    The relative magnitude of weights defines the slope of each softmax
    class probability, and the hyperplanes at which one softmax class
    crosses another represents the boundary between those two classes.
    Boundaries between two dominant classes (those with greatest
    probability for a given region) are called critical boundaries, and
    their normals are the only ones that need to be specified.

    If a space has non-dominant classes (and non-critical boundaries), then
    the weights are slack variables that can take on any values so long as
    the normals of ``all`` dimensions sum up to the 0 vector.

    All normals must sum to zero in each dimension.

    .. image:: img/classes_Softmax.png

    Parameters
    ----------
    weights : array_like, semi-optional
        Weights must be specified as an `M` by `N` matrix, with each row
        representing an `N`-dimensional vector for each of the `M` classes.
    normals : array_like, semi-optional
        Normals must be specified as an `M` by `N` matrix, with one row
        representing each of the `M` classes, and each column representing one
        of `N` dimensions (i.e. `N` = 2 for an x-y cartesian class division).
    bounds : array_like, optional
        Distribution boundaries, in meters, as [xmin, ymin, xmax, ymax].
        Defaults to [-5, -5, 5, 5].
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
    class_cmaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys','RdPu']
    class_colors = ['red', 'blue', 'green', 'orange', 'purple', 'grey','pink']

    def __init__(self, weights=None, normals=None, poly=None, 
                 state_spec='1 x y', bounds=[-5, -5, 5, 5], resolution=0.1,
                 class_labels=None):
        super(SoftMax, self).__init__()
        if weights is not None and normals is not None and poly is not None:
            raise ValueError('One of weights, normals or poly must be '
                             'specified - choose only one!')

        # Plotting attributes
        self.fig = plt.figure(figsize=(14, 8))

        # Define base object attributes
        self.weights = weights
        self.normals = normals
        self.poly = poly
        self.bounds = bounds

        # Create state vector
        self.define_state(state_spec, resolution)

        # Create softmax distributons
        if self.poly is not None:
            self.normals, self.biases = normals_from_polygon(self.poly)

        if self.weights is not None:
            self.num_classes = self.weights.shape[0]
            self.num_params = self.weights.shape[1]
            self.pdfs_from_weights(weights)
        else:
            self.num_classes = self.normals.shape[0] + 1
            self.num_params = self.normals.shape[1]
            self.pdfs_from_normals(self.normals, self.biases)

        # Define superclasses and all class/superclass colors
        self.set_class_labels(class_labels)

    def define_state(self, state_spec, resolution):
        """Create a numeric state vector from a specification.

        Possible specifications:
        * '1'
        * '1 x'
        * 'x'
        * 'y'
        * '1 x y'
        * '1 x y x^2 y^2 2xy'

        Example
        -------
        >>> self.define_state('1 x y')
        >>> print(self.X.shape)
        (101, 101)
        >>> print(self.Y.shape)
        (101, 101)
        >>> print(self.state.shape)
        (10201, 3)

        """

        # <>TODO: Fix for 3D or n-dimensional
        # Define distribution over a gridded state space
        bounds = self.bounds[:]
        self.X, self.Y = np.mgrid[bounds[0]:bounds[2] + resolution:resolution,
                                  bounds[1]:bounds[3] + resolution:resolution]
        self.state = np.dstack((self.X, self.Y))
        self.state = np.reshape(self.state, (self.X.size, 2))
        self.state = np.hstack((np.ones((self.X.size, 1)), self.state))


    def pdfs_from_normals(self, normals, biases=None, interior=True):
        """Create a softmax distributions from a set of normals.

        """
        if biases == None:
            self.biases = np.zeros_like(self.normals[:,[0]])

        #Derive weights from normals
        a = normals.shape[0]
        A = np.roll(np.eye(a,a),1,axis=1) - np.eye(a,a)
        weights = np.dot(np.linalg.pinv(A), normals)

        weights= np.hstack((-biases, weights))
        if interior:
            weights= np.vstack((weights,np.zeros_like(weights[[0],:])))

        self.pdfs_from_weights(weights)
        logging.info("Weights generated from normals:\n {}".format(weights))

    def pdfs_from_weights(self, weights):
        """Create a softmax distributions from a set of weights.

        """
        self.weights = weights

        # Define the softmax normalizer
        sum_ = np.zeros((self.num_classes, self.X.size))
        for i, class_ in enumerate(self.weights):
            sum_[i, :] = np.exp(np.dot(self.state, class_))
        self.normalizer = sum(sum_)  # scalar value

        # Define each class' probability
        self.pdfs = np.zeros((self.X.size, self.num_classes))
        if self.weights is not None:
            for i in range(self.num_classes):
                self.pdfs[:, i] = np.exp(np.dot(self.state, self.weights[i, :]))\
                    / self.normalizer

    def set_class_labels(self, class_labels=None):
        """Sets label and label colors for all classes and subclasses.

        """
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

    def plot(self, plot_classes=True, plot_pdfs=True, plot_poly=False,
             title='SoftMax Classification', **kwargs):
        """Display the class and/or PDF plots of the SoftMax distribution.

        The class plot shows only the critical classes (those that have the
        greatest probability at any given state).

        Parameters
        ----------
        plot_classes : bool, optional
            Plot the critical classes. Defaults to `True`.
        plot_pdfs : bool, optional
            Plot the probability densities. Defaults to `True`.
        plot_poly : bool, optional
            Plot the polygon from which the boundaries are formed. Defaults to 
            `False`.
        **kwargs
            Keyword arguments for ``plot_classes``.
        """
        if plot_classes and plot_pdfs:
            ax1 = self.fig.add_subplot(1, 2, 1)
            ax2 = self.fig.add_subplot(1, 2, 2, projection='3d')
            self._plot_classes(ax1)
            self._plot_pdfs(ax2)
        elif plot_classes:
            self._plot_classes(**kwargs)
        elif plot_pdfs:
            self._plot_pdfs()

        # Create Proxy artists for legend labels
        proxy = [None] * self.num_classes
        for i in range(self.num_classes):
            if self.class_labels[i] not in self.class_labels[:i]:
                proxy_label = self.class_labels[i]
            else:
                proxy_label = "_nolegend_"
            proxy[i] = plt.Rectangle((0, 0), 1, 1, fc=self.class_colors[i],
                                     alpha=0.6, label=proxy_label,)

        plt.legend(handles=proxy, loc='lower center', bbox_to_anchor=(-1.1, -0.175, 2, -0.075),
            mode='expand',borderaxespad=0., ncol=4)
        plt.suptitle(title,fontsize=16)

        # Plot polygon, if possible
        if self.poly is not None and plot_poly and plot_classes:
            patch = PolygonPatch(self.poly, facecolor='none', edgecolor='black',
                             linewidth=3, zorder=2)
            ax1.add_patch(patch)

        plt.show()

    def learn_from_data(self, data):
        """Learn a softmax model from a given dataset.

        """
        pass

    def check_normals(self):
        """Ensures that all normals are correctly formed

        Especially that each dimension of all normals sums to zero.

        """
        # Check the augmented matrix A = [W | n] to see if W and n are linearly dependent
        # Check that all sum to 0
        pass

    def _plot_pdfs(self, ax=None):
        if ax is None:
            ax = self.fig.gca(projection='3d')

        # Define colors
        for i, cmap in enumerate(self.class_cmaps):
            c_index = self.pdfs
            facecolors = plt.get_cmap(cmap)

        # Plot each surface
        for i, color in enumerate(self.class_cmaps):
            prob = self.pdfs[:, i].reshape(self.X.shape)
            cmap = plt.get_cmap(self.class_cmaps[i])
            color = self.class_colors[i]
            surf = ax.plot_surface(self.X, self.Y, prob, rstride=5, cstride=5,
                                   cmap=cmap, alpha=0.8, vmin=0, vmax=1.2,
                                   linewidth=0, edgecolors=color,
                                   antialiased=True, shade=False)

        ax.set_xlim(self.bounds[0], self.bounds[2])
        ax.set_ylim(self.bounds[1], self.bounds[3])

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('Probability P(D=i|X)')
        ax.set_title('Class Probailities')

    def _plot_classes(self, ax=None, plot_poly=False):
        """Plot only the critical classes.

        Critical classes are defined as the classes with highest probability
        for a given state vector `x`.

        """
        # <>TODO: Fix the checking of the state vector to allow, say, 1 x y x^2
        # Plot based on the dimension of the state vector
        if ax is None:
            ax = self.fig.gca()

        if len(self.state[0, :]) == 2:
            self._plot_classes_1D(ax)
        elif len(self.state[0, :]) == 3:
            self._plot_classes_2D(ax)
        elif len(self.state[0, :]) == 4:
            self._plot_classes_3D(ax)
        else:
            raise ValueError('The state vector must be able to be represented '
                             'in 1D, 2D or 3D to be plotted.')

        ax.set_xlim(self.bounds[0], self.bounds[2])
        ax.set_ylim(self.bounds[1], self.bounds[3])

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Critical Classes')

    def _plot_classes_1D(self, ax):
        pass

    def _plot_classes_2D(self, ax):
        # <>TODO: come up with more elegant solution than scatter plot
        # Identify colors of critical classes for each state
        np.set_printoptions(threshold=np.nan)
        max_pdf_indices = np.argmax(self.pdfs, axis=1)
        max_colors = np.take(self.class_colors, max_pdf_indices)
        cc = ColorConverter()
        max_colors_rgb = np.array([cc.to_rgb(_) for _ in max_colors])

        ax.scatter(self.X, self.Y, c=max_colors_rgb, marker='s', s=100,
                   linewidths=0, alpha=1)

        # l_cmap = ListedColormap(self.class_colors, 'Critical Classes')
        # max_pdf_indices = max_pdf_indices.reshape(self.X.shape)
        # ax.imshow(max_pdf_indices,interpolation='none',cmap=l_cmap)

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
    # Turn Polygon into series of vectors
    pts = polygon.exterior.coords[:]
    vects = []
    for i in range(len(pts) - 1):
        vects.append(np.subtract(pts[i + 1], pts[i]))

    # Find unit normal to each vector
    # <>TODO: Extend to 3D
    z = [0, 0, 1]
    normals = (np.cross(vects, z).T / np.linalg.norm(vects, axis=1)).T
    normals = normals[:, :-1]  # removing additional dimension from cross product

    # Define Biases
    biases = np.zeros_like(normals[:,[0]])
    print(biases)
    for i in range(len(pts) - 1):
        midpoint = LineString([pts[i], pts[i + 1]]) \
            .interpolate(0.5, normalized=True)
        biases[i] = midpoint.distance(Point(0,0))
    print(biases)

    logging.info("Normals generated: \n{}".format(normals))
    return normals, biases

def make_simple_softmax():
    weights = np.array([[1, 0.1, 0.1],
                        [1, 0.1, -0.1],
                        [1, -0.1, -0.1]])
    simple = SoftMax(weights)

    simple.plot()


def make_2D_camera(min_view_dist=0.3, max_view_dist=1):
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
    origin = [0, 0]
    a = max_view_dist / np.tan(np.radians(horizontal_view_angle / 2))
    b = min_view_dist / np.tan(np.radians(horizontal_view_angle / 2))
    camera_poly = Polygon([(min_view_dist,b), (max_view_dist,a), (max_view_dist,-a), (min_view_dist,-b)])

    labels = ['Detection', 'No Detection', 'No Detection', 'No Detection',
              'Detection']
    camera = SoftMax(poly=camera_poly)#, class_labels=labels)

    camera.plot(plot_poly=True)


def make_3D_camera():
    # horizontal_view_angle = 57  # from Kinect specifications (degrees)
    # vertical_view_angle = 43  # from Kinect specifications (degrees)

    # # Camera origin
    # origin = [0,0,1] # [x,y,z]
    pass


def make_box(l=1, w=2):
    box_poly = box(0, 0, l, w)
    softmax_box = SoftMax(poly=box_poly)
    softmax_box.plot(plot_poly=False)

def make_poly(poly):
    softmax_poly = SoftMax(poly=poly)
    softmax_poly.plot(plot_poly=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # make_simple_softmax()
    
    # make_2D_camera()
    
    # make_box()
    
    # hexagon = Polygon([(-2,0),(-1,1),(1,1),(2,0),(1,-1),(-1,-1),(-2,0)])
    # make_poly(hexagon)

    #From http://www.mathsisfun.com/geometry/pentagon.html
    pentagon = Polygon([(-1.8996,-0.92915),
                        (-1.395,1.4523),
                        (1.0256,1.7093),
                        (2.018,-0.51393),
                        (0.21001,-2.145),
                        ])
    make_poly(pentagon)

    # poly = Polygon([(-1,-2),(0,1),(1,-2),(-1,-2)])
    # make_poly(poly)

    # t = Polygon([[0,0],[3,3],[6,0],[0,0]])
    # print(normals_from_polygon(t))
