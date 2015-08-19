#!/usr/bin/env python
"""Provides a feasible region for robot and point locations.

Two types of feasible regions are defined: point regions and pose
regions. Point regions are simply all points not contained within
solid physical objects, whereas pose regions are all locations to
which a robot can reach without intersecting with a physical object.

Notes
-----
Only handles static map elements, not dynamic elements.

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
from cops_and_robots.map_tools.map_elements import MapObject
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

    def define_feasible_regions(self, static_elements):
        """Generate the feasible regions from a dictionary of static map elements.

        Parameters
        ----------
        static_elements : dict
            A dictionary of map elements
        """
        # TODO: feasible space should depend on map bounds, not layer bounds
        feasible_space = box(*self.bounds)
        self.point_region = feasible_space
        self.pose_region = feasible_space.buffer(-self.max_robot_radius)

        for element in static_elements:
            # Ignore MapAreas
            if isinstance(element, MapObject):
                self.point_region = self.point_region.difference(element.shape)

                buffered_shape = element.shape.buffer(self.max_robot_radius)
                self.pose_region = self.pose_region.difference(buffered_shape)

    def plot(self, type_="pose", ax=None, alpha=0.5, **kwargs):
        """Plot either the pose or point feasible regions.

        Parameters
        ----------
        type_ : {'pose','point'}
            The type of feasible region to plot.
        ax : figure axis
            The axis to plot the feasible regoin on.
        alpha : int
            Feasible region patch transparency
        **kwargs
            Arguements passed to PolygonPatch
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
