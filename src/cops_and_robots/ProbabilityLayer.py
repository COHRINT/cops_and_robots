#!/usr/bin/env/python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pylab import *
from shapely.geometry import Point


class ProbabilityLayer(object):
    """A probabilistic distribution that represents the x-y position of a given target"""
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

        self.n_particles = 3000
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

    def camera_sensor_update(self):
        """Do a Bayes update of the PDF given the current camera position"""
        pass

if __name__ == '__main__':
    #Run a test probability layer creation        
    pass