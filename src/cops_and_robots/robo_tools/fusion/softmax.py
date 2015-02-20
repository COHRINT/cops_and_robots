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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Seemingly unused, but for 3D plots


class SoftMax(object):
    """Generate a softmax distribution from weights or hyperplane normals.

    The relative magnitude of weights defines the slope of each softmax
    class probability, and the hyperplanes at which one softmax class
    crosses another represents the boundary between those two classes.
    Boundaries between two dominant classes (those with greatest
    probability for a given region) are called critical boundaries, and
    their norms are the only ones that need to be specified.

    If a space has non-dominant classes (and non-critical boundaries), then
    the weights are slack variables that can take on any values so long as
    the norms of ``all`` dimensions sum up to the 0 vector.

    All norms must sum to zero in each dimension.

    .. image:: img/classes_Softmax.png

    Parameters
    ----------
    weights : array_like, semi-optional
        Weights must be specified as an `M` by `N` matrix, with each row
        representing an `N`-dimensional vector for each of the `M` classes.
    norms : array_like, semi-optional
        Norms must be specified as an `M` by `N` matrix, with one row
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
    class_cmaps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys']
    class_colors = ['red', 'blue', 'green', 'orange', 'purple', 'grey']

    def __init__(self, weights=None, norms=None, bounds=[-5, -5, 5, 5],
                 resolution=0.1, class_labels=None):
        super(SoftMax, self).__init__()
        if weights is not None and norms is not None:
            raise ValueError('Both weights and norms specified - choose only '
                             'one!')

        # Define base object attributes
        self.weights = weights
        self.norms = norms
        self.bounds = bounds
        if self.weights is not None:
            self.num_classes = self.weights.shape[0]
            self.num_params = self.weights.shape[1]
        else:
            self.num_classes = self.norms.shape[0]
            self.num_params = self.norms.shape[1]

        # Define superclasses and all class/superclass colors
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

        # <>TODO: Fix for 3D or n-dimensional
        # Define distribution over a gridded space
        self.X, self.Y = np.mgrid[bounds[0]:bounds[2] + resolution:resolution,
                                  bounds[1]:bounds[3] + resolution:resolution]
        self.x = np.dstack((self.X, self.Y))
        self.x = np.reshape(self.x, (self.X.size, 2))
        self.x = np.hstack((np.ones((self.X.size, 1)), self.x))

        # Define the softmax normalizer
        sum_ = np.zeros((self.num_classes, self.X.size))
        for i, class_ in enumerate(self.weights):
            print(sum_)
            sum_[i, :] = np.exp(np.dot(self.x, class_))
        self.normalizer = sum(sum_)  # scalar value

        # Define each class' probability
        self.pdfs = np.zeros((self.X.size, self.num_classes))
        if self.weights is not None:
            for i in range(self.num_classes):
                self.pdfs[:, i] = np.exp(np.dot(self.x, self.weights[i, :]))\
                    / self.normalizer

    def plot(self):
        """Generate a figure for the softmax function.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca(projection='3d')

        proxy = [None] * self.num_classes

        for i, color in enumerate(self.class_cmaps):
            prob = self.pdfs[:, i].reshape(self.X.shape)
            cmap = plt.get_cmap(self.class_cmaps[i])
            color = self.class_colors[i]
            surf = ax.plot_surface(self.X, self.Y, prob, rstride=2, cstride=2,
                                   cmap=cmap, alpha=0.8, vmin=0, vmax=1.2,
                                   linewidth=0, edgecolors=color,
                                   antialiased=True, shade=False)

            # Create Proxy artists for legend labels
            proxy[i] = plt.Rectangle((0, 0), 1, 1, fc=self.class_colors[i],
                                     alpha=0.6)

            # fig.colorbar(surf)

        ax.set_xlim(self.bounds[0], self.bounds[2])
        ax.set_ylim(self.bounds[1], self.bounds[3])
        # ax.set_zlim(-0.01, 1.01)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('Probability P(D=i|X)')
        ax.set_title('Softmax distribution')

        ax.legend(proxy, self.class_labels)
        # plt.tight_layout()
        plt.show()

    def learn_from_data(self, data):
        """Learn a softmax model from a given dataset.

        """
        pass

    def check_norms(self):
        """Ensures that all norms are correctly formed

        Especially that each dimension of all norms sums to zero.

        """
        pass


def make_simple_softmax():
    # weights = np.array([[1, 0.1, 0.1],
    #                     [1, 0.1, -0.1],
    #                     [1, -0.1, -0.1]])
    weights = np.array([[2.8721, -0.2223, -0.7769],
                        [-2.6089, 1.0582, 0.5460]])
    simple = SoftMax(weights)

    # weights = np.array([[-4.58589689939030,6.50488972803049,-0.000634799985029515],
    #                    [-0.635589026753982,-242.553237254745,-423.697666826221],
    #                    [-0.638808964339006,-242.014980120127,422.728842454847],
    #                    [0,0,0]
    #                    ])
    # labels = ['ND','ND','D']
    # simple = SoftMax(weights, class_labels=labels)

    simple.plot()


def make_2D_camera(min_view_dist=0, max_view_dist=1):
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
    view_slope = math.tan(math.radians(horizontal_view_angle / 2))

    # Define class weights
    weights = np.array([[1, view_slope, 0],
                        [1, 0, 0],
                        [1, -view_slope, 0],
                        [1, 0, min_view_dist],
                        [1, 0, max_view_dist],
                        ])
    labels = ['No Detection', 'No Detection', 'No Detection', 'No Detection',
              'Detection']
    camera = SoftMax(weights, class_labels=labels)
    camera.plot()


def make_3D_camera():
    # horizontal_view_angle = 57  # from Kinect specifications (degrees)
    # vertical_view_angle = 43  # from Kinect specifications (degrees)

    # # Camera origin
    # origin = [0,0,1] # [x,y,z]
    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    make_simple_softmax()
    # make_2D_camera()