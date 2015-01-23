#!/usr/bin/env python
"""Provides a feasible region for robot locations.

Two types of feasible regions are defined: point regions and pose 
regions. Point regions are simply all points not contained within 
solid physical objects, whereas pose regions are all locations to 
which a robot can reach without intersecting with a physical object.

Note:
    Currently, the feasible layer uses the shape layer to generate
    the ideal feasible regions. However, a more realistic approach 
    would use the occupancy layer. This will likely be the approach 
    taken in future versions.

Note:
    Currently, the feasible layer us the same for all robots. This is 
    not realistic, and will likely change in future versions.

Required Knowledge:
    This module and its classes needs to know about the following 
    other modules in the cops_and_robots parent module:
        1. ``layer`` for generic layer parameters and functions.
        2. ``shape_layer`` to generate the feasible regions.
"""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

from matplotlib.colors import cnames
from shapely.geometry import box,Polygon

from cops_and_robots.map_tools.layer import Layer
from cops_and_robots.map_tools.shape_layer import ShapeLayer

class FeasibleLayer(Layer):
    """A representation of feasible map regions.

    A polygon (or collection of polygons) that represent feasible 
    regions of the map. Feasible can be defined as either feasible robot 
    poses or unoccupied space.

    :param max_robot_radius: the maximum radius in [m] of all robots.
    :type max_robot_radius: positive float.
    """
    def __init__(self,max_robot_radius=0.20,**kwargs):
        super(FeasibleLayer, self).__init__(**kwargs)
        self.max_robot_radius = max_robot_radius #[m] conservative estimate of robot size

        self.point_region = None
        self.pose_region = None
        self.define_feasible_regions()               

    def define_feasible_regions(self,shape_layer=None):
        """Generate the feasible regions from a given shape layer.

        :param shape_layer: the collection of shapes on the map.
        :type shape_layer: ShapeLayer.
        """
        if not shape_layer:
            shape_layer = ShapeLayer(self.bounds)
        
        feasible_space = box(*self.bounds)
        for obj_ in shape_layer.shapes.values():
            self.point_region = feasible_space.difference(obj_.shape)

            buffered_shape = obj_.shape.buffer(self.max_robot_radius)
            self.pose_region = feasible_space.difference(buffered_shape)

    def plot(self,type_="pose"):
        """Plot either the pose or point feasible regions.

        :param type_: the type of feasible region to plot.
        :type type_: String of 'pose' or 'point'.
        :returns: <>TODO
        :rtype:<>TODO
        """
        if type_ is "pose":
            p = self.pose_region.plot()
        else:
            p = self.point_region.plot()
        return p
