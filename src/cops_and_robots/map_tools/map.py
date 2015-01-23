#!/usr/bin/env python
"""Collects all the information a robot has of its environment.

A map has three different parts: its properties, its elements, and its 
layers. Properties define key intrinsics of the map (such as its 
boundaries). Elements define the collections of physical objects on 
the map (robots, walls, etc.). Layers (meant to be toggled for 
visibility) represent perspectives on the map: where the map's 
elements are, what probability distributions are associated with a 
given element, etc.

Required Knowledge:
    This module and its classes needs to know about the following 
    other modules in the cops_and_robots parent module:
        1. ``map_obj`` to represent map elements (like walls).
        2. ``shape_layer`` to collect map elements.
        3. ``occupancy_layer`` to represent a grid of occupied space.
        4. ``feasible_layer`` for unoccupied and reachable space.
        5. ``probability_layer`` to represent continuous probability.
        6. ``particle_layer`` to represent discrete probability points.
"""

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
from shapely.geometry import Point
from descartes.patch import PolygonPatch


from cops_and_robots.map_tools.map_obj import MapObj
from cops_and_robots.map_tools.shape_layer import ShapeLayer
from cops_and_robots.map_tools.occupancy_layer import OccupancyLayer
from cops_and_robots.map_tools.feasible_layer import FeasibleLayer
from cops_and_robots.map_tools.probability_layer import ProbabilityLayer
from cops_and_robots.map_tools.particle_layer import ParticleLayer

class Map(object):
    """Environment map composed of multiple elements and layers.

    :param mapname: the name of the map.
    :type mapname: String.
    :param bounds: map boundaries as [x_min,y_min,x_max,y_max].
    :type bounds: list of floats.
    """

    def __init__(self,mapname,bounds):
        #Define map properties
        self.mapname = mapname
        self.bounds = bounds #[x_min,y_min,x_max,y_max] in [m] useful area
        self.outer_bounds = [i * 1.1 for i in self.bounds] 
        self.origin = [0,0] #in [m]
        self.fig = plt.figure(1,figsize=(10,8)) 


        #Define map elements
        self.objects = {} #Dict of object.name : MapObj object, for all
                          #dynamic and static map objects (not robbers or cops)
        self.cops = {} #Dict of cop.name : MapObj object, for each cop
        self.robbers = {} #Dict of robber.name : MapObj object, for each robber

        #Define layers
        self.shapes = ShapeLayer(bounds=bounds) #Single ShapeLayer object
        self.occupancy = OccupancyLayer(bounds=bounds) #Single OccupancyLayer object
        self.feasible = FeasibleLayer(bounds=bounds) #Single FeasibleLayer object        
        self.particle = {} #Dict of robber.name : ParticleLayer objects, 
                           #one per robber
        self.probability = {} #Dict of robber.name : ProbabilityLayer objects, 
                              #one per robber, plus one for avg. robber pose
        
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

    def add_cop(self,cop_obj):
        """Add a dynamic ``Robot`` cop from the Map

        :param cop_obj: Map object
        :type cop: MapObj.
        """
        # self.shapes.add_obj(cop)
        self.cops[cop_obj.name] = cop_obj

    def rem_cop(self,cop_name):
        """Remove a dynamic ``Robot`` cop from the Map by its name

        :param cop_name: String
        """
        # self.shapes.rem_obj(cop_name)
        del self.cops[cop_name]

    def add_robber(self,robber,n_particles=2000):
        """Add a dynamic ``Robot`` robber from the Map

        :param robber: Robot
        """
        self.robbers[robber.name] = robber
        self.particle[robber.name] = ParticleLayer()
        # self.probability[robber.name] = ProbabilityLayer(self.bounds,robber)
        #<>update average probability layer

    def rem_robber(self,robber_name):
        """Remove a dynamic ``Robot`` robber from the Map by its name

        :param robber_name: String
        """
        self.shapes.rem_obj(robber_name)
        del self.robbers[robber_name]
        del self.probability[robber_name]
        #<>Update average probability layer

    def plot(self,robber_name="combined",plot_zones=True,feasible_region="pose"):
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
        

    def setup_plot(self):
        """Create the initial plot for the animation.
        """
        if len(self.robbers) == 1:
            self.ax_list = [self.fig.add_subplot(111)]
        else:
            f,ax_list = plt.subplots(len(self.robbers/2,2,num=1))
            self.ax_list = [ax_list for sublist in ax_list[:] for ax_list in sublist]

        for ax in self.ax_list:
            ax.set_xlim([self.bounds[0],self.bounds[2]])
            ax.set_ylim([self.bounds[1],self.bounds[3]])

        #Plot static elements
        for i,robber in enumerate(self.robbers.values()):
            ax = self.ax_list[i]
            self.shapes.plot(plot_zones=False,ax=ax)

        #plot first round of dynamic elements
        # self.animated_plots = []
        # for i,ax in enumerate(self.ax_list):
        #     self.animated_plots.append({})

        # for i,particle_filter in enumerate(self.particle_filter.values()):
        #     ax = self.ax_list[i]
        #     self.animated_plots[i]['particles'] = particle_filter.plot(ax=ax)
        #     self.animated_plots[i]['sensor'] = self.sensor.plot(ax=ax,plot_zones=False,color=self.sensor.default_color,alpha=self.sensor.alpha)
        #     self.animated_plots[i]['cop'] = self.plot()

        # p_particles = [p['particles'] for p in self.animated_plots]
        # p_sensor = [p['sensor'] for p in self.animated_plots]
        # p_cop = [p['cop'] for p in self.animated_plots]
        # p_sensor = p_sensor[0]
        # print((p_particles,p_sensor,p_cop))
        
        #Define cop path
        self.movement_path = Line2D((0,0),(0,0),linewidth=2,alpha=0.4,color=cnames['green'])
        ax.add_line(self.movement_path)

        #Define cop patch
        simple_poly = Point((0,0)).buffer(0.01)
        self.cop_patch = PolygonPatch(simple_poly)
        ax.add_patch(self.cop_patch)

        #Define particle filter
        arbitrary_particle_layer = next(self.particle.itervalues())
        init_particles = np.zeros((arbitrary_particle_layer.n_particles,3))
        self.scat = ax.scatter(init_particles[:,0],
                               init_particles[:,1],
                               c=init_particles[:,2],
                               cmap=arbitrary_particle_layer.cmap,
                               s=arbitrary_particle_layer.particle_size,
                               lw=arbitrary_particle_layer.line_weight,
                               alpha=arbitrary_particle_layer.alpha,
                               marker='.',
                               vmin=0,
                               vmax=1
                               )

        
        #Define camera patch
        self.camera_patch = PolygonPatch(simple_poly)
        ax.add_patch(self.camera_patch)

        #Define robber patch
        self.robber_patch = PolygonPatch(simple_poly)
        ax.add_patch(self.robber_patch)

        return self.cop_patch,self.movement_path,self.camera_patch,self.scat,self.robber_patch

    def animation_stream(self):
        """Generate new values for the animation plot, based on an 
        updated model of the world.
        """
        #Initialize packet values to zeros
        simple_poly = Point((0,0)).buffer(0.01)
        particles = np.zeros((2000,3))
        cop_shape = simple_poly
        cop_path = [0,0]
        camera_shape = simple_poly
        robber_shape = simple_poly

        packet = {}
        for robber in self.robbers:
            packet[robber] = (cop_shape,cop_path,camera_shape,particles,robber_shape)

        while True:
            for pkt in packet.values():

                cop_shape,cop_path,camera_shape,particles,robber_shape = pkt

                #Update cop patch
                self.cop_patch.remove()
                self.cop_patch = PolygonPatch(cop_shape, facecolor=cnames['green'], alpha=0.9, zorder=2)
                self.ax_list[0].add_patch(self.cop_patch)

                #Update movement path
                self.movement_path.set_data(cop_path)

                #Update sensor patch
                self.camera_patch.remove()
                self.camera_patch = PolygonPatch(camera_shape, facecolor=cnames['yellow'], alpha=0.3, zorder=2)
                self.ax_list[0].add_patch(self.camera_patch)

                #Update Particle Filter
                colors = particles[:,2]*600
                self.scat.set_array(colors)
                self.scat.set_offsets(particles[:,0:2])

                #Update robber patch
                self.robber_patch.remove()
                self.robber_patch = PolygonPatch(robber_shape, facecolor=cnames['orange'], alpha=0.9, zorder=2)
                self.ax_list[0].add_patch(self.robber_patch)
                
                # if self.found_target['Roy']:
                #     for i,particle in enumerate(self.particle_filter['Roy'].particles):
                #         if self.particle_filter['Roy'].particle_probs[i] == 1:
                #             target_particle = particle
                #     self.ax_list[0].scatter(target_particle[0],target_particle[1],marker='x',s=1000,color=cnames['darkred'])


                # sensor_patch = PolygonPatch(self.sensor.shape, facecolor=self.sensor.default_color, alpha=self.sensor.alpha, zorder=2)
                # # cop_patch = PolygonPatch(self.shape, facecolor=self.default_color, alpha=self.alpha, zorder=2)
                # cop_path = self.cop_patch.get_path()
                # cop_path.vertices = [map(sum,zip(a,self.pose[0:2]))for a in cop_path.vertices]
                # self.cop_patch = patches.PathPatch(cop_path, facecolor='orange', lw=2)

                # for i,particle_filter in enumerate(self.particle_filter.values()):
                #     self.animated_plots[i]['particles'].set_array(particle_filter.particle_probs)
                #     self.animated_plots[i]['sensor'] = sensor_patch
                #     # self.animated_plots[i]['cop'] = cop_patch

                # p_particles = [p['particles'] for p in self.animated_plots]
                # p_sensor = [p['sensor'] for p in self.animated_plots]
                # p_cop = [p['cop'] for p in self.animated_plots]
                # yield self.movement_path,self.scat,self.cop_patch,self.sensor_patch
                packet = yield self.scat

def set_up_fleming():
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
    
    return fleming

if __name__ == "__main__":
    # fleming = set_up_fleming()

    # fleming.plot('Roy')
    # fleming.probability['Roy'].update(fleming.objects['Wall_4'],'back')
    # fleming.plot('Roy')
    pass