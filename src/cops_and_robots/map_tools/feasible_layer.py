#!/usr/bin/env python
"""Provides a feasible region for robot and point locations.

Two types of feasible regions are defined: point regions and pose
regions. Point regions are simply all points not contained within
solid physical objects, whereas pose regions are all locations to
which a robot can reach without intersecting with a physical object.

Note
----
    Currently, the feasible layer uses the shape layer to generate
    the ideal feasible regions. However, a more realistic approach
    would use the occupancy layer. This will likely be the approach
    taken in future versions.

    The feasible layer is the same for all robots. This is
    not realistic, and will likely change in future versions.

"""
__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import logging

from shapely.geometry import box

from cops_and_robots.map_tools.layer import Layer
from cops_and_robots.map_tools.shape_layer import ShapeLayer
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt
from matplotlib.colors import cnames


class FeasibleLayer(Layer):
    """A representation of feasible map regions.

    A polygon (or collection of polygons) that represent feasible
    regions of the map. Feasible can be defined as either feasible robot
    poses or unoccupied space.

    .. image:: img/classes_Feasible_Layer.png

    Parameters
    ----------
    max_robot_radius : float, optional
        The maximum radius of a circular approximation to the robot, used
        to determine the feasible pose regions.
    **kwargs
        Arguments passed to the ``Layer`` superclass.

    """
    def __init__(self, max_robot_radius=0.20, **kwargs):
        super(FeasibleLayer, self).__init__(**kwargs)
        self.max_robot_radius = max_robot_radius  # [m] conservative estimate

        self.point_region = None
        self.pose_region = None
        # self.define_feasible_regions()

    def define_feasible_regions(self, shape_layer=None):
        """Generate the feasible regions from a given shape layer.

        Parameters
        ----------
        shape_layer : ShapeLayer, optional
            The shape layer from which to generate the feasible regions. If
            no layer is provided, the entire map is deemed feasible.
        """
        if shape_layer is None:
            shape_layer = ShapeLayer(bounds=self.bounds)

        feasible_space = box(*self.bounds)
        self.point_region = feasible_space
        self.pose_region = feasible_space

        for obj_ in shape_layer.shapes.values():
            self.point_region = self.point_region.difference(obj_.shape)

            buffered_shape = obj_.shape.buffer(self.max_robot_radius)
            self.pose_region = self.pose_region.difference(buffered_shape)

    def plot(self, type_="pose", ax=None, alpha=0.5, plot_spaces=False, **kwargs):
        """Plot either the pose or point feasible regions.

        Parameters
        ----------
        type_ : {'pose','point'}
            The type of feasible region to plot.
        """
        if type_ == "pose":
            p = self.pose_region
        else:
            p = self.point_region

        if ax is None:
            ax = plt.gca()

        ax.set_xlim([self.bounds[0], self.bounds[2]])
        ax.set_ylim([self.bounds[1], self.bounds[3]])

        patch = PolygonPatch(p, facecolor=cnames['black'],
                             alpha=alpha, zorder=2, **kwargs)
        ax.add_patch(patch)
        plt.show()

        # return patch
