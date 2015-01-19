#!/usr/bin/env python
"""MODULE_DESCRIPTION"""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pylab import *
from shapely.geometry import Point
from cops_and_robots.ParticleFilter import *


class ProbabilityLayer(object):
    """A probabilistic distribution that represents the x-y position of a given target"""
    def __init__(self,bounds,target,cell_size=0.2,visible=True):
        self.target = target #Name of target to be tracked by this layer
        self.bounds = bounds #[xmin,ymin,xmax,ymax] in [m]
        self.visible = visible
        self.cell_size = cell_size #in [m/cell]

        xlin = np.linspace(bounds[0],bounds[2],100)
        ylin = np.linspace(bounds[1],bounds[3],100)
        self.X, self.Y = np.meshgrid(xlin,ylin)
        self.pos = np.empty(self.X.shape + (2,))
        self.pos[:, :, 0] = self.X
        self.pos[:, :, 1] = self.Y

        self.MAP = [0, 0] #[m] point of maximum a posteriori

        self.prob = stats.multivariate_normal([0,0], [[10,0], [0,10]])
        
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

        self.MAP = np.mean(self.kept_particles,axis=0)


    def camera_sensor_update(self):
        """Do a Bayes update of the PDF given the current camera position"""
        pass

if __name__ == '__main__':
    #Run a test probability layer creation        
    pass