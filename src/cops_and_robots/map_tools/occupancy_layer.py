#!/usr/bin/env python
"""Provides a gridded sense of which map regions are empty or not.

An occupancy grid takes the map boundaries, creates a grid (with 
arbitrary cell size) and gives each cell a value: 0 for unoccupied 
cells, 1 for occupied cells, and None for cells that have not yet 
been explored.

Note:
    The Bernoulli distribution (0 or 1) per cell is ideal, and a true 
    occupancy grid would include a continuous probability per cell of 
    its likelihood of occupancy. This will likely be added in future
    versions.

Note:
    The occupancy grid currently considers only static objects. This 
    will also likely change in the future versions.

Required Knowledge:
    This module and its classes needs to know about the following 
    other modules in the cops_and_robots parent module:
        1. ``layer`` for generic layer parameters and functions.
        2. ``shape_layer`` to generate the occupied regions.
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

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from shapely.geometry import box

from cops_and_robots.map_tools.layer import Layer
from cops_and_robots.map_tools.shape_layer import ShapeLayer

class OccupancyLayer(Layer):
    """Generate an occupancy grid from static map objects.

    Gridded occupancy layer for the map, translating euclidean coordinates 
    to grid cells. Each cell has a probability of occupancy from 0 to 1.

    :param cell_size: the size of each cell in the occupancy grid.
    :type cell_size: positive float.
    """

    def __init__(self,cell_size=0.05,**kwargs):
        super(OccupancyLayer, self).__init__(**kwargs)
        self.cell_size = cell_size #[m/cell]

        self.grid = []
        x,y = self.bounds[0],self.bounds[1]
        c = self.cell_size
        while x + c <= self.bounds[2]:
            while y + c <= self.bounds[3]:
                #Create cells with grid centered on (0,0)
                cell = box(x, y, x+c, y+c)
                self.grid.append(cell)
                y = y+c
            x = x+c
            y = self.bounds[1]

        self.n_cells = len(self.grid)

        self.grid_occupancy = 0.5 * np.ones((self.n_cells,1), dtype=np.int)
    
    def add_obj(self,map_obj):
        """Fill the cells for a given map_obj.

        :param map_obj: the map object considered.
        :type map_obj: MapObj.
        """
        for i,cell in enumerate(self.grid):
            if map_obj.shape.intersects(cell):
                self.grid_occupancy[i] = 1

    def rem_obj(self,map_obj):
        """Empty the cells for a given map_obj.

        :param map_obj: the map object considered.
        :type map_obj: MapObj.
        """
        for i,cell in enumerate(self.grid):
            if map_obj.shape.intersects(cell):
                self.grid_occupancy[i] = 0

    def occupancy_from_shape_layer(self,shape_layer):
        """Create an occupancy grid from an entire shape layer.

        :param shape_layer: the collection of shapes on the map.
        :type shape_layer: ShapeLayer
        """
        pass

    def plot(self):
        """Plot the occupancy grid as a pseudo colormesh.

        :returns: the scatter pseudo colormesh data.
        :rtype: QuadMesh.
        """
        xsize = self.bounds[2] - self.bounds[0]
        ysize = self.bounds[3] - self.bounds[1]
        grid = self.grid_occupancy.reshape(xsize/self.cell_size,ysize/self.cell_size)
        X,Y = np.mgrid[0:grid.shape[0]:1,0:grid.shape[1]:1]
        X,Y = (X*self.cell_size, Y*self.cell_size)
        p = plt.pcolormesh(X, Y, grid, cmap=cm.Greys)
        #<>TODO: add in cell borders!
        return p

if __name__ == '__main__':
    #Run a test occupancy layer creation
    pass