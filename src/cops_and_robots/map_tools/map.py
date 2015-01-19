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

from pylab import *
from matplotlib.colors import cnames

import matplotlib.pyplot as plt
import random

from cops_and_robots.robo_tools.fusion.particle_filter import ParticleFilter
from cops_and_robots.robo_tools.robber import Robber
from cops_and_robots.map_tools.MapObj import MapObj
from cops_and_robots.map_tools.ShapeLayer import ShapeLayer
from cops_and_robots.map_tools.OccupancyLayer import OccupancyLayer
from cops_and_robots.map_tools.ProbabilityLayer import ProbabilityLayer
from cops_and_robots.map_tools.FeasibleLayer import FeasibleLayer


class Map(object):
    """Environment map composed of several layers, including an occupancy layer and a probability layer for each robber.

    :param mapname: String
    :param bounds: [x_max,y_max]"""
    def __init__(self,mapname,bounds):
        self.mapname = mapname
        self.bounds = bounds #[x_min,y_min,x_max,y_max] in [m] useful area
        self.outer_bounds = [i * 1.1 for i in self.bounds] #[x_max,y_max] in [m] total area to be shown
        self.origin = [0,0] #in [m]
        self.shapes = ShapeLayer(bounds) #Single ShapeLayer object
        self.occupancy = OccupancyLayer(bounds) #Single OccupancyLayer object
        self.feasible = FeasibleLayer(bounds) #Single FeasibleLayer object        
        self.particles = {} #Dict of robber.name : ParticleFilter objects, one per robber
        self.probability = {} #Dict of robber.name : ProbabilityLayer objects, one per robber, plus one for avg. robber pose
        self.probability["average"] = ProbabilityLayer(self.bounds,"average")
        self.objects = {} #Dict of object.name : MapObj object, for all dynamic and static map objects (not robbers or cop)
        self.cops = {} #Dict of cop.name : Cop object, for each cop
        self.robbers = {} #Dict of robber.name : Robot object, for each robber

    def add_obj(self,map_obj):
        """Append a static ``MapObj`` to the Map

        :param map_obj: MapObj object
        """
        self.objects[map_obj.name] = map_obj
        self.occupancy.add_obj(map_obj)
        self.shapes.add_obj(map_obj)
        self.feasible.define_feasible_regions(self.shapes)
        #<>TODO: Update probability layer

    def rem_obj(self,map_obj_name):
        """Remove a ``MapObj`` from the Map by its name

        :param map_obj_name: String
        """
        self.shapes.rem_obj(map_obj_name)
        self.feasible.define_feasible_regions(self.shapes)
        self.occupancy.rem_obj(self.objects[map_obj_name])
        del self.objects[map_obj_name]
        #<>TODO: Update probability layer

    def add_cop(self,cop):
        """Add a dynamic ``Robot`` cop from the Map

        :param cop: Robot
        """
        # self.shapes.add_obj(cop)
        self.cops[cop.name] = cop

    def rem_cop(self,cop_name):
        """Remove a dynamic ``Robot`` cop from the Map by its name

        :param cop_name: String
        """
        # self.shapes.rem_obj(cop_name)
        del self.cops[cop_name]

    def add_robber(self,robber):
        """Add a dynamic ``Robot`` robber from the Map

        :param robber: Robot
        """
        self.shapes.add_obj(robber,all_objects=False)
        self.robbers[robber.name] = robber
        self.particles[robber.name] = ParticleFilter(self.bounds,robber)
        self.probability[robber.name] = ProbabilityLayer(self.bounds,robber)
        #<>update average probability layer

    def rem_robber(self,robber_name):
        """Remove a dynamic ``Robot`` robber from the Map by its name

        :param robber_name: String
        """
        self.shapes.rem_obj(robber_name)
        del self.robbers[robber_name]
        del self.probability[robber_name]
        #<>Update average probability layer

    def plot(self,robber_name="average",plot_zones=True,feasible_region="pose"):
        """Generate one or more probability and occupancy layer

        :param robber_name: String
        """
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)

        #Plot shape layer
        if self.shapes.visible:
            self.shapes.plot(plot_zones)

        #Plot occupancy layer
        if self.occupancy.visible:
            occ = self.occupancy.plot()
        
        #Plot feasible layer
        # if self.feasible.visible:
        #     self.feasible.plot(feasible_region)

        #Plot particle layer
        if self.particles[robber_name].visible:
            self.particles[robber_name].plot()

        # #Plot probability layer
        # if self.probability[robber_name].visible:
        #     prob, cb = self.probability[robber_name].plot()
        #     ax.grid(b=False) 

        # #Plot cop
        # if self.cop.visible:
        #     self.cop.plot()

        # #Plot robbers
        # for robber in self.robbers:
        #     if robber.visible:
        #         robber.plot()

        #Plot robbers   

        # #Plot relative position polygons
        # if plot_zones:
        #     for map_obj in self.objects:
        #         if self.objects[map_obj].has_zones:
        #             self.objects[map_obj].add_to_plot(ax,include_shape=False,include_zones=True)
        
        # #Plot particles
        # # plt.scatter(self.probability[robber_name].particles[:,0],self.probability[robber_name].particles[:,1],marker='x',color='r')
        # if plot_particles:
        #     if len(self.probability[robber_name].kept_particles) > 0:
        #         plt.scatter(self.probability[robber_name].kept_particles[:,0],self.probability[robber_name].kept_particles[:,1],marker='x',color='w')

        # #Plot robot position
        #<>TODO

        plt.xlim([self.outer_bounds[0], self.outer_bounds[2]])
        plt.ylim([self.outer_bounds[1], self.outer_bounds[3]])
        plt.show()

        return ax
        
def set_up_fleming(robber_names = ['Leon','Pris','Roy','Zhora']):    
    #Make vicon field space object
    net_w = 0.2 #[m] Netting width
    field_w = 10 #[m] field width
    netting = MapObj('Netting',[field_w+net_w,field_w+net_w],has_zones=False) 
    field = MapObj('Field',[field_w,field_w],has_zones=False) 

    #Make wall objects
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
    bounds = [-field_w/2,-field_w/2,field_w/2,field_w/2]
    fleming = Map('Fleming',bounds)

    fleming.add_obj(netting)
    fleming.add_obj(field)
    fleming.rem_obj('Field')
    fleming.rem_obj('Netting')
    # fleming.objects['Netting'].visible=False

    #Add walls to map
    for wall in walls:
        fleming.add_obj(wall)
    
    #Add robbers (randomly positioned) to map
    for robber_name in robber_names:
        x = random.uniform(fleming.bounds[0],fleming.bounds[2])
        y = random.uniform(fleming.bounds[1],fleming.bounds[3])
        theta = random.uniform(0,359)
        pose = (x,y,theta)
        robber = Robot(robber_name,pose=pose,default_color=cnames['darkorange'])
        fleming.add_robber(robber)

    #Add cop to map
    deckard = Robot('Deckard')
    fleming.add_cop(deckard)

    return fleming

if __name__ == "__main__":
    fleming = set_up_fleming()

    fleming.plot('Roy')
    fleming.probability['Roy'].update(fleming.objects['Wall_4'],'back')
    fleming.plot('Roy')
    # fleming.probability['Roy'].update(fleming.objects['Wall_1'],'front')
    # fleming.probability['Roy'].update(fleming.objects['Wall_1'],'front')
    # fleming.plot('Roy')
    # fleming.probability['Roy'].update(fleming.objects['Wall_2'],'back')
    # fleming.probability['Roy'].update(fleming.objects['Wall_2'],'back')
    # fleming.plot('Roy')
    # fleming.probability['Roy'].update(fleming.objects['Wall_2'],'right')
    # fleming.probability['Roy'].update(fleming.objects['Wall_2'],'right')    
    # fleming.plot('Roy')