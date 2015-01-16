#!/usr/bin/env/python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from pylab import *
from shapely.geometry import box

class OccupancyLayer(object):
    """Gridded occupancy layer for the map, translating euclidean coordinates to grid cells. Each cell has a probability of occupancy from 0 to 1."""

    def __init__(self, bounds, cell_size=0.05,visible=True):
        self.visible = visible
        self.bounds = bounds #[xmin,ymin,xmax,ymax] in [m]
        self.cell_size = cell_size #[m/cell]

        self.grid = []
        x,y = bounds[0],bounds[1]
        c = self.cell_size
        while x + c <= bounds[2]:
            while y + c <= bounds[3]:
                #Create cells with grid centered on (0,0)
                cell = box(x, y, x+c, y+c)
                self.grid.append(cell)
                y = y+c
            x = x+c
            y = bounds[1]

        self.n_cells = len(self.grid)

        self.grid_occupancy = 0.5 * np.ones((self.n_cells,1), dtype=np.int)
    
    def add_obj(self,map_obj):
        for i,cell in enumerate(self.grid):
            if map_obj.shape.intersects(cell):
                self.grid_occupancy[i] = 1

    def rem_obj(self,map_obj):
        for i,cell in enumerate(self.grid):
            if map_obj.shape.intersects(cell):
                self.grid_occupancy[i] = 0

    def plot(self):
        xsize = self.bounds[2] - self.bounds[0]
        ysize = self.bounds[3] - self.bounds[1]
        grid = self.grid_occupancy.reshape(xsize/self.cell_size,ysize/self.cell_size)
        X,Y = np.mgrid[0:grid.shape[0]:1,0:grid.shape[1]:1]
        X,Y = (X*self.cell_size, Y*self.cell_size)
        p = plt.pcolor(X, Y, grid, cmap=cm.Greys)

if __name__ == '__main__':
    #Run a test occupancy layer creation
    pass