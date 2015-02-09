#!/usr/bin/env python
"""Provides a grouping of all map objects.

The shape layer accounts for all map objects, grouping them both as a
dictionary of individual objects as well as an accumulation of all
objects in one.

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``layer`` for generic layer parameters and functions.
        2. ``map_obj`` to represent parts of the shape layer.
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

from shapely.geometry import MultiPolygon, Polygon

from cops_and_robots.map_tools.layer import Layer


class ShapeLayer(Layer):
    """docstring for ShapeLayer

    :param alpha: transparency of all objects on this layer.
    :type alpha: float from 0 to 1.
    """
    def __init__(self, alpha=0.9, **kwargs):
        super(ShapeLayer, self).__init__(alpha=alpha, **kwargs)
        self.shapes = {}  # Dict of MapObj.name : Map object, for each shape
        self.all_shapes = Polygon()

    def add_obj(self, map_obj, update_all_objects=True):
        """Add a map object to the shape layer.

        :param map_obj: the object to be added to the layer.
        :type map_obj: MapObj.
        :param update_all_objects: whether to update the aggregation of
            all objects.
        :type update_all_objects: bool.
        """
        self.shapes[map_obj.name] = map_obj
        if update_all_objects:
            self.update_all()

    def rem_obj(self, map_obj_name, all_objects=True):
        """Remove a map object from the shape layer.

        :param map_obj_name: the object name to be removed.
        :type map_obj_name: String.
        :param update_all_objects: whether to update the aggregation of
            all objects.
        :type update_all_objects: bool.
        """
        del self.shapes[map_obj_name]
        if all_objects:
            self.update_all()

    def update_all(self):
        """Update the aggregation of all shapes into one object.
        """
        all_shapes = []
        for obj_ in self.shapes.values():
            all_shapes.append(obj_.shape)
        self.all_shapes = MultiPolygon(all_shapes)

    def plot(self, plot_zones=True, **kwargs):
        """Plot all visible map objects (and their zones, if visible).

        :param plot_zones: whether or not to plot map object zones.
        :type plot_zones: bool.
        """
        for shape in self.shapes.values():
            if shape.visible:
                shape.plot(plot_zones=plot_zones, alpha=self.alpha, **kwargs)
