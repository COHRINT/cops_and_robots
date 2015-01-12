#!/usr/bin/env/python
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from shapely.geometry import box

class OccupancyLayer(object):
    """Gridded occupancy layer for the map, translating euclidean coordinates to grid cells. Each cell has a probability of occupancy from 0 to 1."""

    def __init__(self, map_, cell_size=0.05):
        self.xbound = map_.outer_bounds[0] #[m]
        self.ybound = map_.outer_bounds[1] #[m]
        self.cell_size = cell_size #[m/cell]

        self.area = [self.xbound,self.ybound] #[m]

        self.grid = []
        x,y = 0,0
        c = self.cell_size
        while x + c <= self.xbound:
            while y + c <= self.ybound:
                #Create cells with grid centered on (0,0)
                cell = box(x-self.xbound/2, y-self.ybound/2, x+c-self.xbound/2, y+c-self.ybound/2)
                self.grid.append(cell)
                y = y+c
            x = x+c
            y = 0

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
        grid = self.grid_occupancy.reshape(self.xbound/self.cell_size-1,self.ybound/self.cell_size-1)
        X,Y = np.mgrid[0:grid.shape[0]:1,0:grid.shape[1]:1]
        X,Y = (X*self.cell_size - self.xbound/2, Y*self.cell_size - self.ybound/2)
        p = plt.pcolor(X, Y, grid, cmap=cm.Greys)

if __name__ == '__main__':
    #Run a test occupancy layer creation
    pass