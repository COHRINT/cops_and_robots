#!/usr/bin/env python
"""Provides a grouping of all map objects.

The shape layer accounts for all map objects, grouping them both as a
dictionary of individual objects as well as an accumulation of all
objects in one.

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
    """A layer object containing all fixed map objects.

    Parameters
    ----------
    **kwargs
        Arguments for the ``Layer`` superclass.

    """
    def __init__(self, **kwargs):
        super(ShapeLayer, self).__init__(**kwargs)
        self.alpha = 0.9
        self.shapes = {}  # Dict of MapObj.name : Map object, for each shape
        self.all_shapes = Polygon()

    def add_obj(self, map_obj, update_all_objects=True):
        """Add a map object to the shape layer.

        Parameters
        ----------
        map_obj : MapObj
            The object to be added.
        update_all_objects : bool, optional
            `True` to update the aggregation of all objects. Default is `True`.

        """
        self.shapes[map_obj.name] = map_obj
        if update_all_objects:
            self.update_all()

    def rem_obj(self, map_obj_name, all_objects=True):
        """Remove a map object from the shape layer.

        Parameters
        ----------
        map_obj_name : str
            The name of the object to be removed.
        update_all_objects : bool, optional
            `True` to update the aggregation of all objects. Default is `True`.
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

        Parameters
        ----------
        plot_zones : bool, optional
            `True` to plot the zones of the map objects. Default is `True`.

        """
        for shape in self.shapes.values():
            if shape.visible:
                shape.plot(plot_zones=plot_zones, alpha=self.alpha, **kwargs)
