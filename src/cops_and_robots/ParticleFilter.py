#!/usr/bin/env/python
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from shapely.geometry import Polygon


class ParticleFilter(object):
    """Particle-based representation of target locations"""
    def __init__(self,bounds,target,cell_size=0.2,n_particles = 3000,visible=True):
        super(ParticleFilter, self).__init__()
        self.target = target #MapObj of target for specific map layer
        self.bounds = bounds
        self.visible = visible
        self.cell_size = cell_size #in [m/cell]

        xlin = np.linspace(bounds[0],bounds[2],100)
        ylin = np.linspace(bounds[1],bounds[3],100)
        self.X, self.Y = np.meshgrid(xlin,ylin)
        self.pos = np.empty(self.X.shape + (2,))
        self.pos[:, :, 0] = self.X
        self.pos[:, :, 1] = self.Y

        self.n_particles = n_particles
        self.particles = stats.multivariate_normal([0,0], [[10,0], [0,10]]).rvs(size=self.n_particles)
        self.particle_probs = np.ones(n_particles)/n_particles
        
    def plot(self,particle_size=200, ax=None,alpha=0.3):
        if ax == None:
            ax = plt.gca()
        cm = plt.cm.get_cmap('jet')
        p = ax.scatter(self.particles[:,0],self.particles[:,1],marker='.',c=self.particle_probs,s=particle_size,cmap=cm,lw=0,alpha=alpha,vmin=0,vmax=1)
        # cb = plt.colorbar(p)    
        return p
    
    def update(self,camera,target_pose,relative_str=""):
        self.particle_probs = self.camera_update(camera,target_pose)
        # if relative_str is not "":
        #     self.human_update(map_obj,relative_str)

        # self.resample()

    def camera_update(self,camera,target_pose):
        particle_probs = camera.detect(self.particles,self.particle_probs,target_pose)
        return particle_probs

    def human_update(self,map_obj,relative_str):
        #<>break human out into seperate class

        #check particles within box denoted by relative string
        poly = map_obj.zones_by_label[relative_str]
        for i,particle in enumerate(self.particles):
            point = Point(particle)
            if poly.contains(point):
                kept_particles.append(particle)
                self.particle_probs[i] = 1
            else:
                self.particle_probs[i] = 0

        self.kept_particles = np.asarray(kept_particles)

    def resample(self):
        '''Sample or resample distribution to generate new particles'''

        #sample distribution
        #<>Constrain particle limits to map limits
        #<>Always resample self.n_particles
        self.particles = self.prob.rvs(size=self.n_particles)
        print(self.particles)

        #recalculate particles
        self.particle_probs = [1/self.n_particles for i in range(0,self.n_particles)]

if __name__ == '__main__':
    #Run a test probability layer creation        
    pass        