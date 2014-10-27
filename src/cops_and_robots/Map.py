#!/usr/bin/env/python
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy import stats
from pylab import *

class Map(object):
    """Environment map composed of several layers, including an occupancy layer and a probability layer for each target."""
    def __init__(self,mapname,bounds):
        self.mapname = mapname
        self.bounds = bounds #[x_max,y_max] in [m] total area to be shown
        self.origin = [0,0] #in [m]
        self.occupancy = None #Single OccupancyLayer object
        self.probability = None #Dict of ProbabilityLayer objects
        self.objects = []
        self.targets = []

    def add_obj(self,map_obj):
        """Append a ``MapObj`` to the Map

        :param map_obj: MapObj object
        """
        self.objects.append(map_obj)
        self.occupancy.add_obj(map_obj)
        #<>TODO: Update probability layer

    def rem_obj(self,map_obj_name):
        """Remove a ``MapObj`` from the Map by its name

        :param map_obj_name: String
        """
        for map_obj in self.objects:
            if map_obj_name == map_obj.name:
                print('Removing \'', map_obj_name, '\' from the map.')
                self.objects.remove(map_obj)
                self.occupancy.rem_obj(map_obj)
                break
        else:
            print('No object named \'', map_obj_name,'\' found!')
        #<>TODO: Update probability layer

    def plot_map(self):
        self.occupancy.plot()
        self.probability['Roy'].plot()


class MapObj(object):
    """docstring for MapObj"""
    def __init__(self,name,shape,pose=[0,0,0],centroid=[0,0,0]):
        self.name = name #sting identifier
        self.shape = shape #[l,w] in [m] length and width of a rectangular object   
        #self.shape      = shape     #nx2 array containing all (x,y) vertices for the object shape
        self.pose = pose #[x,y,theta] in [m] coordinates of the centroid in the global frame
        self.centroid = centroid  #(x,y,theta) [m] coordinates of the centroid in the object frame

    def __str___(self):
        return "%s is located at (%d,%d), pointing at %d" % (self.name, self.centroid['x'],self.centroid['y'],self.centroid['theta'])

class OccupancyLayer(object):
    """Gridded occupancy layer for the map, translating euclidean coordinates to grid cells. Each cell has a probability of occupancy from 0 to 1."""

    def __init__(self, map_, cell_size=0.05):
        self.xbound = map_.bounds[0] #[m]
        self.ybound = map_.bounds[1] #[m]
        self.cell_size = cell_size #[m/cell]

        self.area = [self.xbound,self.ybound] #[m]
        self.grid = 0.5 * np.ones((self.xbound/self.cell_size, self.ybound/self.cell_size), dtype=np.int)
        self.n_cells = self.grid.size
        
    def add_obj(self,map_obj):

        #TODO: find grid cells more than %50 occupied by object
        
        #Find containing box around object [x_0 y_0 len width]
        # a = max(map_obj.w,map_obj.l) #[m] box side length
        # self.box = [[a/2+map_obj.x, a/2+map_obj.y],[a, a]]

        x,y,theta = map_obj.pose
        #CHEAP HACK
        if theta == 0:
            [l,w] = [map_obj.shape[1],map_obj.shape[0]]
        else:
            [l,w] = [map_obj.shape[0],map_obj.shape[1]]

        min_x_cell = int((-w/2 + x)/self.cell_size) + self.grid.shape[0]//2
        max_x_cell = int((w/2 + x)/self.cell_size ) + self.grid.shape[0]//2
        min_y_cell = int((-l/2 + y)/self.cell_size) + self.grid.shape[1]//2
        max_y_cell = int((l/2 + y)/self.cell_size)  + self.grid.shape[1]//2

        #Horizontal flip hack
        # tmp = max_x_cell
        # max_x_cell = self.grid.shape[0] - min_x_cell
        # min_x_cell = self.grid.shape[0] - tmp

        self.grid[min_x_cell:max_x_cell,min_y_cell:max_y_cell] = 1

    def rem_obj(self,map_obj):
        x,y,theta = map_obj.pose
        #CHEAP HACK
        if theta == 0:
            [l,w] = [map_obj.shape[1],map_obj.shape[0]]
        else:
            [l,w] = [map_obj.shape[0],map_obj.shape[1]]

        min_x_cell = int((-w/2 + x)/self.cell_size) + self.grid.shape[0]//2
        max_x_cell = int((w/2 + x)/self.cell_size ) + self.grid.shape[0]//2
        min_y_cell = int((-l/2 + y)/self.cell_size) + self.grid.shape[1]//2
        max_y_cell = int((l/2 + y)/self.cell_size)  + self.grid.shape[1]//2

        self.grid[min_x_cell:max_x_cell, min_y_cell:max_y_cell] = 0

    def plot(self):
        X,Y = np.mgrid[0:self.grid.shape[0]:1,0:self.grid.shape[1]:1]
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1, 1, 1)

        p = ax.pcolor(X, Y, self.grid, cmap=cm.Greys)

        plt.show()

class ProbabilityLayer(object):
    """Probability heatmap for expected target location."""
    def __init__(self,map_,target,cell_size=0.2):
        self.target = target #MapObj of target for specific map layer
        self.xbound = map_.bounds[0] #[m]
        self.ybound = map_.bounds[1] #[m]
        self.cell_size = cell_size #in [m]

        self.X, self.Y = np.mgrid[0:self.xbound:self.cell_size, 0:self.ybound:self.cell_size]
        self.pos = np.empty(self.X.shape + (2,))
        self.pos[:, :, 0] = self.X
        self.pos[:, :, 1] = self.Y

        x, y, theta = [0,0,0]
        prior = 1/self.X.size * stats.uniform.pdf(np.ones(self.X.shape))
        self.prob = prior
        

    def plot(self):
        
        fig = plt.figure(figsize=(20,6))

        ax = fig.add_subplot(1, 2, 1)

        p = ax.pcolor(self.X, self.Y, self.prob, cmap=cm.hot)#, vmin=abs(prior).min(), vmax=abs(prior).max())
        cb = fig.colorbar(p, ax=ax)    

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        p = ax.plot_surface(self.X, self.Y, self.prob, rstride=1, cstride=1, cmap=cm.hot,linewidth=0, antialiased=True)
        cb = fig.colorbar(p, ax=ax)    

        plt.show()

        
    def update(self,pose,str_):
        #use str_ to determine the type of update (front, back, etc.)
        likelihood = stats.multivariate_normal([0,0], [[10,0], [0,10]])
        self.prob = self.prob * likelihood.pdf(self.pos)

        
if __name__ == "__main__":
    #Make vicon field space
    net_w = 0.2 #[m] Netting width
    field_w = 10 #[m] field area
    netting = MapObj('Netting',[field_w,field_w]) 
    field = MapObj('Field',[field_w-net_w,field_w-net_w]) 

    #Make walls
    l = 1.2192 #[m] wall length
    w = 0.1524 #[m] wall width
    x = [0, l, (l*1.5+w/2), 2*l, l,   (l*2.5+w/2), l, (l*1.5+w/2)]
    y = [0, l, (l*1.5-w/2), 2*l, 3*l, (l*1.5-w/2), 0, (-l*0.5+w/2)]
    theta = [0, 0, math.pi/2,   0,   0,   math.pi/2,   0, math.pi/2]

    walls = []
    for i in range(0,len(x)):
        name = 'Wall' + str(i)
        pose = [x[i],y[i],theta[i]]
        wall = MapObj(name, [l,w], pose)
        walls.append(wall)      

    #Create Fleming map
    fleming = Map('Fleming',[field_w*1.2,field_w*1.2])
    fleming.occupancy = OccupancyLayer(fleming,0.05)

    fleming.add_obj(netting)
    fleming.add_obj(field)
    fleming.rem_obj('Field')

    for wall in walls:
        fleming.add_obj(wall)
    
    fleming.targets = ['Leon','Pris','Roy','Zhora']
    fleming.probability = {tar: ProbabilityLayer(fleming,tar) for tar in fleming.targets}
    
    fleming.plot_map()

    fleming.probability['Roy'].update(walls[0].pose,'Front')

    fleming.plot_map()