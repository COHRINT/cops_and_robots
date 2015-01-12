#!/usr/bin/env/python
import matplotlib.pyplot as plt
from pylab import *
from cops_and_robots.MapObj import *
from cops_and_robots.Robot import *
from cops_and_robots.OccupancyLayer import *
from cops_and_robots.ProbabilityLayer import *


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

    def plot_map(self,target_name="",plot_occ=True,plot_prob=True,
                 plot_particles=True,plot_zones=True,plot_robot=False):
        """Generate one or more probability and occupancy layer

        :param target_name: String
        """
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)

        #Plot occupancy layer
        if plot_occ:
            occ = self.occupancy.plot()
        
        #Plot probability layer
        if plot_prob:
            if target_name == "":
                for tar_name in self.targets:
                    prob, cb = self.probability[tar_name].plot()
            else:
                prob, cb = self.probability[target_name].plot()
            ax.grid(b=False)    

        #Plot relative position polygons
        if plot_zones:
            for map_obj in self.objects:
                if self.objects[map_obj].has_zones:
                    self.objects[map_obj].add_to_plot(ax,include_shape=False,include_zones=True)
        
        #Plot particles
        # plt.scatter(self.probability[target_name].particles[:,0],self.probability[target_name].particles[:,1],marker='x',color='r')
        if plot_particles:
            if len(self.probability[target_name].kept_particles) > 0:
                plt.scatter(self.probability[target_name].kept_particles[:,0],self.probability[target_name].kept_particles[:,1],marker='x',color='w')

        #Plot robot position
        #<>TODO

        plt.xlim([-self.outer_bounds[0]/2, self.outer_bounds[0]/2])
        plt.ylim([-self.outer_bounds[1]/2, self.outer_bounds[1]/2])
        plt.show()

        return ax
        
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
    fleming.occupancy = OccupancyLayer(fleming)

    fleming.add_obj(netting)
    fleming.add_obj(field)
    fleming.rem_obj('Field')

    #Add walls to map
    for wall in walls:
        fleming.add_obj(wall)
    
    #Add targets to map
    tar_names = ['Leon','Pris','Roy','Zhora']
    for tar_name in tar_names:
        tar = Robot(tar_name)
        fleming.add_tar(tar)

    return fleming

if __name__ == "__main__":
    fleming = set_up_fleming()

    fleming.plot_map('Roy')
    fleming.probability['Roy'].update(fleming.objects['Wall_4'],'back')
    fleming.plot_map('Roy')
    # fleming.probability['Roy'].update(fleming.objects['Wall_1'],'front')
    # fleming.probability['Roy'].update(fleming.objects['Wall_1'],'front')
    # fleming.plot_map('Roy')
    # fleming.probability['Roy'].update(fleming.objects['Wall_2'],'back')
    # fleming.probability['Roy'].update(fleming.objects['Wall_2'],'back')
    # fleming.plot_map('Roy')
    # fleming.probability['Roy'].update(fleming.objects['Wall_2'],'right')
    # fleming.probability['Roy'].update(fleming.objects['Wall_2'],'right')    
    # fleming.plot_map('Roy')