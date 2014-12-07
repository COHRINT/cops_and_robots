#!/usr/bin/env/python
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy import stats
from pylab import *
import Robot #Change this to robbers
from MapObj import MapObj
from shapely.geometry import Polygon,Point,box

class Map(object):
    """Environment map composed of several layers, including an occupancy layer and a probability layer for each target.

    :param mapname: String
    :param bounds: [x_max,y_max]"""
    def __init__(self,mapname,bounds):
        self.mapname = mapname
        self.bounds = bounds #[x_max,y_max] in [m] useful area
        self.outer_bounds = [i * 1.1 for i in self.bounds] #[x_max,y_max] in [m] total area to be shown
        self.origin = [0,0] #in [m]
        self.occupancy = None #Single OccupancyLayer object
        self.probability = {} #Dict of target.name : ProbabilityLayer objects
        self.objects = {} #Dict of object.name : MapObj object
        self.targets = {} #Dict of target.name : Robot object
        self.cop = {} #Cop object

    def add_obj(self,map_obj):
        """Append a static ``MapObj`` to the Map

        :param map_obj: MapObj object
        """
        self.objects[map_obj.name] = map_obj
        self.occupancy.add_obj(map_obj)
        #<>TODO: Update probability layer

    def rem_obj(self,map_obj_name):
        """Remove a ``MapObj`` from the Map by its name

        :param map_obj_name: String
        """
        self.occupancy.rem_obj(self.objects[map_obj_name])
        del self.objects[map_obj_name]
        #<>TODO: Update probability layer

    def add_tar(self,target):
        """Add a dynamic ``Robot`` target from the Map

        :param target: Robot
        """
        self.targets[target.name] = target
        self.probability[target.name] = ProbabilityLayer(self,target)

    def rem_tar(self,target_name):
        """Remove a dynamic ``Robot`` target from the Map by its name

        :param target_name: String
        """
        del self.targets[target_name]
        #<>Update probability layer

    def plot_map(self,target_name=""):
        """Generate one or more probability and occupancy layer

        :param target_name: String
        """
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)

        #Plot occupancy layer
        occ = self.occupancy.plot()
        
        #Plot probability layer
        if target_name == "":
            for tar_name in self.targets:
                prob, cb = self.probability[tar_name].plot()
        else:
            prob, cb = self.probability[target_name].plot()
        ax.grid(b=False)    

        #Plot relative position polygons
        for map_obj in self.objects:
            if self.objects[map_obj].has_zones:
                self.objects[map_obj].add_to_plot(ax,include_shape=False,include_zones=True)
        
        #Plot particles
        # plt.scatter(self.probability[target_name].particles[:,0],self.probability[target_name].particles[:,1],marker='x',color='r')
        if len(self.probability[target_name].kept_particles) > 0:
            plt.scatter(self.probability[target_name].kept_particles[:,0],self.probability[target_name].kept_particles[:,1],marker='x',color='w')

        plt.xlim([-self.outer_bounds[0]/2, self.outer_bounds[0]/2])
        plt.ylim([-self.outer_bounds[1]/2, self.outer_bounds[1]/2])
        plt.show()


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


class ProbabilityLayer(object):
    """Probability heatmap for expected target location."""
    def __init__(self,map_,target,cell_size=0.2):
        self.target = target #MapObj of target for specific map layer
        self.xbound = map_.bounds[0] #[m]
        self.ybound = map_.bounds[1] #[m]
        self.cell_size = cell_size #in [m/cell]

        xlin = np.linspace(-self.xbound/2,self.xbound/2,100)
        ylin = np.linspace(-self.ybound/2,self.ybound/2,100)
        self.X, self.Y = np.meshgrid(xlin,ylin)
        self.pos = np.empty(self.X.shape + (2,))
        self.pos[:, :, 0] = self.X
        self.pos[:, :, 1] = self.Y

        self.n_particles = 1000
        self.particles = []
        self.kept_particles = [] #TEST STUB
        self.particle_weights = np.zeros(1000)

        self.ML = [0, 0] #[m] point of maximum likelihood

        # uni_x = stats.uniform.pdf(self.X, loc=0, scale=self.xbound)
        # uni_y = stats.uniform.pdf(self.Y, loc=0, scale=self.ybound)
        # self.prob = np.dot(uni_x, uni_y) #uniform prior
        # prior = 1/self.X.size * stats.uniform.pdf(np.ones(self.X.shape))
        # self.prob = prior
        # self.prob = stats.multivariate_normal([0,0], [[100000,0], [0,1000]])
        self.prob = stats.multivariate_normal([0,0], [[10,0], [0,10]])
        # self.prob = self.prob.pdf(self.pos)
        
    def plot(self):
        p = plt.pcolor(self.X, self.Y, self.prob.pdf(self.pos), cmap=cm.jet, alpha=0.7)#, vmin=abs(prior).min(), vmax=abs(prior).max())
        cb = plt.colorbar(p)    
        return p, cb
    
    def update(self,map_obj,relative_str):
        '''Updates particle filter values and generate new probabilty layer'''

        #generate particles
        self.resample()
        kept_particles = []

        #check particles within box denoted by relative string
        poly = map_obj.zones_by_label[relative_str]
        for i,particle in enumerate(self.particles):
            point = Point(particle)
            if poly.contains(point):
                kept_particles.append(particle)
                self.particle_weights[i] = 1
            else:
                self.particle_weights[i] = 0

        self.kept_particles = np.asarray(kept_particles)

        if len(kept_particles) == 0:
            mean = [0,0]
            var = [[10,0], [0,10]]
        else:
            mean = np.mean(self.kept_particles,axis=0)
            var = np.var(self.kept_particles,axis=0)

        #generate gaussian from new particles
        #TEST STUB
        prob_update = stats.multivariate_normal(mean,var)
        self.prob = prob_update
        # self.prob = self.prob * likelihood.pdf(self.pos)
        # self.prob = np.dot(self.prob, likelihood)

        self.ML = np.mean(self.kept_particles,axis=0)

    def resample(self):
        '''Sample distribution to generate new particles'''

        #sample distribution
        #<>Constrain particle limits to map limits
        #<>Always resample self.n_particles
        self.particles = self.prob.rvs(size=self.n_particles)

        #reweight particles
        self.particle_weights = [1/self.n_particles for i in range(0,self.n_particles)]
        
    
def set_up_fleming():    
    #Make vicon field space
    net_w = 0.2 #[m] Netting width
    field_w = 10 #[m] field area
    netting = MapObj('Netting',[field_w+net_w,field_w+net_w],has_zones=False) 
    field = MapObj('Field',[field_w,field_w],has_zones=False) 

    #Make walls
    l = 1.2192 #[m] wall length
    w = 0.1524 #[m] wall width
    wall_shape = [l,w]    
    x = [0, l, (l*1.5+w/2), 2*l, l,   (l*2.5+w/2), l, (l*1.5+w/2)]
    y = [0, l, (l*1.5-w/2), 2*l, 3*l, (l*1.5-w/2), 0, (-l*0.5+w/2)]
    theta = [0, 0, 90,   0,   0,   90,   0, 90]

    walls = []
    for i in range(0,len(x)):
        name = 'Wall_' + str(i)
        pose = (x[i],y[i],theta[i])
        wall = MapObj(name, wall_shape, pose)
        
        walls.append(wall)      

    #Create Fleming map
    bounds = [field_w,field_w]
    fleming = Map('Fleming',bounds)
    fleming.occupancy = OccupancyLayer(fleming,0.05)

    fleming.add_obj(netting)
    fleming.add_obj(field)
    fleming.rem_obj('Field')

    #Add walls to map
    for wall in walls:
        fleming.add_obj(wall)
    
    #Add targets to map
    tar_names = ['Leon','Pris','Roy','Zhora']
    for tar_name in tar_names:
        tar = Robot.Robot(tar_name)
        fleming.add_tar(tar)

    return fleming

if __name__ == "__main__":
    fleming = set_up_fleming()

    fleming.plot_map('Roy')
    fleming.probability['Roy'].update(fleming.objects['Wall_0'],'left')
    fleming.plot_map('Roy')
    fleming.probability['Roy'].update(fleming.objects['Wall_1'],'front')
    fleming.probability['Roy'].update(fleming.objects['Wall_1'],'front')
    fleming.plot_map('Roy')
    fleming.probability['Roy'].update(fleming.objects['Wall_2'],'back')
    fleming.probability['Roy'].update(fleming.objects['Wall_2'],'back')
    fleming.plot_map('Roy')
    fleming.probability['Roy'].update(fleming.objects['Wall_2'],'right')
    fleming.probability['Roy'].update(fleming.objects['Wall_2'],'right')    
    fleming.plot_map('Roy')