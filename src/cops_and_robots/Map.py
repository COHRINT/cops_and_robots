#!/usr/bin/env/python
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy import stats
from pylab import *
import Robot #Change this to robbers

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
            if self.objects[map_obj].has_poly:
                plt.plot(self.objects[map_obj].front_poly[:,0],self.objects[map_obj].front_poly[:,1])
        
        #Plot particles
        # plt.scatter(self.probability[target_name].particles[:,0],self.probability[target_name].particles[:,1],marker='x',color='r')
        if len(self.probability[target_name].kept_particles) > 0:
            plt.scatter(self.probability[target_name].kept_particles[:,0],self.probability[target_name].kept_particles[:,1],marker='x',color='w')

        plt.xlim([-self.outer_bounds[0]/2, self.outer_bounds[0]/2])
        plt.ylim([-self.outer_bounds[1]/2, self.outer_bounds[1]/2])
        plt.show()


class MapObj(object):
    """Generate one or more probability and occupancy layer

        :param target_name: String
        """
    def __init__(self,name,shape,pose=[0,0,0],centroid=[0,0,0],has_poly=0):
        self.name = name #sting identifier
        self.shape = shape #[l,w] in [m] length and width of a rectangular object   
        #self.shape      = shape     #nx2 array containing all (x,y) vertices for the object shape
        self.pose = pose #[x,y,theta] in [m] coordinates of the centroid in the global frame
        self.centroid = centroid  #(x,y,theta) [m] coordinates of the centroid in the object frame
        self.has_poly = has_poly

        #Relative polygons
        if has_poly:
            l,w = (shape[0], shape[1])
            spread = [1.5, 5] #trapezoidal spread from anchor points
            if l > w: 
                x = [-l/2, -spread[0]*l/2, spread[0]*l/2, l/2, -l/2]
                x = [i + pose[0] for i in x]
                y = [w/2, spread[1]*w, spread[1]*w, w/2, w/2]
                y = [i + pose[1] for i in y]
            else:
                y = [-w/2, -spread[0]*w/2, spread[0]*w/2, w/2, -w/2]
                y = [i + pose[1] for i in y]
                x = [l/2, spread[1]*l, spread[1]*l, l/2, l/2]
                x = [i + pose[0] for i in x]
            self.front_poly = np.array([list(i) for i in list(zip(x,y))])
            self.back_poly = []
            self.left_poly = []
            self.right_poly = []
        else:
            self.front_poly = np.array([])
            self.back_poly = []
            self.left_poly = []
            self.right_poly = []

    def __str___(self):
        return "%s is located at (%d,%d), pointing at %d" % (self.name, self.centroid['x'],self.centroid['y'],self.centroid['theta'])


class OccupancyLayer(object):
    """Gridded occupancy layer for the map, translating euclidean coordinates to grid cells. Each cell has a probability of occupancy from 0 to 1."""

    def __init__(self, map_, cell_size=0.05):
        self.xbound = map_.outer_bounds[0] #[m]
        self.ybound = map_.outer_bounds[1] #[m]
        self.cell_size = cell_size #[m/cell]

        self.area = [self.xbound,self.ybound] #[m]
        self.grid = 0.5 * np.ones((self.xbound/self.cell_size+1, self.ybound/self.cell_size+1), dtype=np.int)
        self.n_cells = self.grid.size
        
    def add_obj(self,map_obj):

        #TODO: find grid cells more than %50 occupied by object
        
        #Find containing box around object [x_0 y_0 len width]
        # a = max(map_obj.w,map_obj.l) #[m] box side length
        # self.box = [[a/2+map_obj.x, a/2+map_obj.y],[a, a]]

        x,y,theta = map_obj.pose
        w,l = map_obj.shape

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
        w,l = map_obj.shape

        min_x_cell = int((-w/2 + x)/self.cell_size) + self.grid.shape[0]//2
        max_x_cell = int((w/2 + x)/self.cell_size ) + self.grid.shape[0]//2
        min_y_cell = int((-l/2 + y)/self.cell_size) + self.grid.shape[1]//2
        max_y_cell = int((l/2 + y)/self.cell_size)  + self.grid.shape[1]//2

        self.grid[min_x_cell:max_x_cell, min_y_cell:max_y_cell] = 0

    def plot(self):
        X,Y = np.mgrid[0:self.grid.shape[0]:1,0:self.grid.shape[1]:1]
        X,Y = (X*self.cell_size - self.xbound/2, Y*self.cell_size - self.ybound/2)
        p = plt.pcolor(X, Y, self.grid, cmap=cm.Greys)


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
        if relative_str == 'front':
            path = Path(map_obj.front_poly)
            for i,particle in enumerate(self.particles):
                if path.contains_point(particle):
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
    netting = MapObj('Netting',[field_w+net_w,field_w+net_w]) 
    field = MapObj('Field',[field_w,field_w]) 

    #Make walls
    l = 1.2192 #[m] wall length
    w = 0.1524 #[m] wall width
    x = [0, l, (l*1.5+w/2), 2*l, l,   (l*2.5+w/2), l, (l*1.5+w/2)]
    y = [0, l, (l*1.5-w/2), 2*l, 3*l, (l*1.5-w/2), 0, (-l*0.5+w/2)]
    theta = [0, 0, math.pi/2,   0,   0,   math.pi/2,   0, math.pi/2]

    walls = []
    for i in range(0,len(x)):
        name = 'Wall' + str(i)
        #CHEAP HACK
        pose = [x[i],y[i],theta[i]]
        if theta[i] == 0:
            wall = MapObj(name, [l,w], pose, has_poly=1)
        else:
            wall = MapObj(name, [w,l], pose, has_poly=1)
        
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
    fleming.probability['Roy'].update(fleming.objects['Wall0'],'front')
    fleming.plot_map('Roy')
    fleming.probability['Roy'].update(fleming.objects['Wall1'],'front')
    fleming.plot_map('Roy')
    fleming.probability['Roy'].update(fleming.objects['Wall2'],'front')
    fleming.plot_map('Roy')
    fleming.probability['Roy'].update(fleming.objects['Wall2'],'front')
    fleming.plot_map('Roy')