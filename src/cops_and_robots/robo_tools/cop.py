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
import random
import math

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.transforms as tf
from matplotlib.colors import cnames

from shapely.geometry import LineString,Point
from shapely import affinity
from descartes.patch import PolygonPatch

from cops_and_robots.robo_tools.robot import Robot
from cops_and_robots.map_tools.Map import Map,set_up_fleming
from cops_and_robots.map_tools.ParticleFilter import ParticleFilter
from cops_and_robots.robo_tools.fusion.Camera import Camera
from cops_and_robots.robo_tools.fusion.Human import Human
from cops_and_robots.map_tools.MapObj import MapObj


class Cop(Robot):
    """docstring for Cop"""
    def __init__(self,pose=[0,0.5,0]):
        #Superclass and compositional attributes
        Robot.__init__(self,pose=pose,name="Deckard",default_color=cnames['darkgreen'])
        self.map = set_up_fleming(['Roy'])
        self.particle_filter = {}
        self.found_target = {}
        for robber in self.map.robbers.values():
            self.particle_filter[robber.name] = ParticleFilter(self.map.bounds,robber)
            self.found_target[robber.name] = False
        self.sensor = Camera(pose)
        self.human = Human()

        #Movement attributes
        self.move_distance = 0.2 #[m] per time step
        self.rotate_distance = 15 #[deg] per time step
        self.update_rate = 1 #[Hz]
        self.goal_pose = [0.2,0.5,0] #[x,y,theta] in [m]
        self.pose_history = [] #list of [x,y,theta] in [m]
        self.pose_history.append(self.pose)
        self.check_last_n = 50 #number of poses to look at before assuming stuck
        self.stuck_distance = 0.1 #[m] distance traveled in past self.check_last_n to assume stuck
        self.approximate_allowance = 0.5 #[m]
        self.rotation_allowance = 5 #[deg]
        self.num_goals = None #number of goals to reach. None for infinite
        
        #Animation attributes
        self.fig = plt.figure(1,figsize=(10,8)) 
        self.stream = self.animation_stream()

    def translate_to_pose(self):
        """Move the robot's x,y positions to the next 

        """        
        next_point = self.movement_line.interpolate(self.move_distance)
        self.pose[0:2] = (next_point.x,next_point.y)
        self.movement_line = LineString((self.pose[0:2], self.goal_pose[0:2]))
        print("Translated to {}".format(["{:.2f}".format(a) for a in self.pose]))

    def rotate_to_pose(self):       
        #Rotate ccw or cw
        angle_diff = self.goal_pose[2] - self.pose[2]
        rotate_ccw = (abs(angle_diff) < 180) and  self.goal_pose[2] > self.pose[2] or (abs(angle_diff) > 180) and  self.goal_pose[2] < self.pose[2]
        if rotate_ccw: #rotate ccw
            next_angle = min(self.rotate_distance,abs(angle_diff))
        else:
            next_angle = -min(self.rotate_distance,abs(angle_diff))
        print('Next angle: {:.2f}'.format(next_angle))
        self.pose[2] = (self.pose[2] + next_angle) % 360
        print("Rotated to {}".format(["{:.2f}".format(a) for a in self.pose]))

    def find_goal_pose(self):
        max_particles = []
        max_prob = 0
        for particle_filter in self.particle_filter.values():
            particle_filter.update(self.sensor,self.map.robbers['Roy'].pose)
            max_ = max(particle_filter.particle_probs)
            if max_ > max_prob:
                max_prob = max_
            for i,particle in enumerate(particle_filter.particles):
                if particle_filter.particle_probs[i] == max_prob:
                    max_particles.append(particle)

        #Select randomly from max_particles
        self.goal_pose[0:2] = random.choice(max_particles)
        self.movement_line = LineString((self.pose[0:2], self.goal_pose[0:2]))
        self.goal_pose[2] = math.atan2(self.goal_pose[1] - self.pose[1],self.goal_pose[0] - self.pose[0]) #[rad]
        self.goal_pose[2] = math.degrees(self.goal_pose[2]) % 360

        print("Moving to goal {}".format(["{:.2f}".format(a) for a in self.goal_pose]))

    def sensor_update(self):
        for key in self.particle_filter:
            probs = self.sensor.update(self.pose,self.map.shapes.all_shapes,self.particle_filter[key],self.map.robbers['Roy'].pose)
            self.particle_filter[key].particle_probs = probs
            if max(probs) == 1:
                self.found_target[key] = True

    def check_if_stuck(self):
        check_last_n = min(self.check_last_n,len(self.pose_history))
        
        distance_travelled = 0
        last_poses = self.pose_history[-check_last_n:]
        for i,pose in enumerate(last_poses):
            if i > 0:
                dist = math.sqrt((last_poses[i-1][0] - pose[0]) ** 2 + (last_poses[i-1][1] - pose[1]) ** 2)
                distance_travelled += dist
        print('Travelled {:.2f}m in last {}'.format(distance_travelled,check_last_n))
        self.stuck = distance_travelled < self.stuck_distance


    def update(self,i):
        #Check if close to goal
        goal_pt = Point(self.goal_pose[0:2])
        identification_viewcone = affinity.scale(self.sensor.shape,xfact=0.5,yfact=0.5,origin=self.sensor.view_pose[0:2])
        close_enough = identification_viewcone.intersects(goal_pt)

        #Check if stuck
        self.check_if_stuck()
        
        if close_enough or (self.stuck and i>10) or self.found_target['Roy']:
            if self.stuck:
                print('Stuck!')
            self.find_goal_pose()

        #Check if heading in right direction, and move
        properly_rotated = abs(self.pose[2] - self.goal_pose[2]) < self.rotation_allowance
        
        if properly_rotated:
            self.translate_to_pose()
        else:
            self.rotate_to_pose()

        self.pose_history.append(self.pose[:])

        self.sensor_update()

        return next(self.animation_stream())

    def setup_plot(self):
        if len(self.map.robbers.values()) == 1:
            self.ax_list = [self.fig.add_subplot(111)]
        else:
            f,ax_list = plt.subplots(len(self.map.robbers.values())/2,2,
                num=1)
            self.ax_list = [ax_list for sublist in ax_list[:] for ax_list in sublist]

        for ax in self.ax_list:
            lim = 5
            ax.set_xlim([-lim,lim])
            ax.set_ylim([-lim,lim])

        #plot static elements
        for i,robber in enumerate(self.map.robbers.values()):
            ax = self.ax_list[i]
            self.map.shapes.plot(plot_zones=False,ax=ax)

        #plot first round of dynamic elements
        self.animated_plots = []
        for i,ax in enumerate(self.ax_list):
            self.animated_plots.append({})

        self.sensor.alpha = 0.6
        self.alpha = 0.8
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
        
        #Define particle filter
        self.scat = self.particle_filter['Roy'].plot()
        self.scat.set_array(100*np.random.random(self.particle_filter['Roy'].n_particles))

        #Define movement path
        self.movement_path = Line2D([0,self.pose[0]],[0,self.pose[1]],color=self.default_color,linewidth=2,alpha=0.4)
        ax.add_line(self.movement_path)

        #Define cop patch
        self.cop_patch = PolygonPatch(self.shape, facecolor=self.default_color, alpha=self.alpha, zorder=2)
        ax.add_patch(self.cop_patch)
        
        #Define sensor patch
        self.sensor_patch = PolygonPatch(self.sensor.shape, facecolor=self.sensor.default_color, alpha=self.sensor.alpha, zorder=2)
        ax.add_patch(self.sensor_patch)

        return self.scat,self.movement_path,self.cop_patch,self.sensor_patch

    def animation_stream(self):
        #while not self.found_target.values().count(False)
        while True:
            #Update Particle Filter
            colors = self.particle_filter['Roy'].particle_probs*self.particle_filter['Roy'].n_particles/3
            self.scat.set_array(colors)            
            if self.found_target['Roy']:
                for i,particle in enumerate(self.particle_filter['Roy'].particles):
                    if self.particle_filter['Roy'].particle_probs[i] == 1:
                        target_particle = particle
                self.ax_list[0].scatter(target_particle[0],target_particle[1],marker='x',s=1000,color=cnames['darkred'])

            #Update movement path
            line_data = self.movement_path.get_data()
            if len(line_data[0]) > self.check_last_n:
                line_data = [a[-self.check_last_n:] for a in line_data]
            x,y = line_data[0],line_data[1]
            self.movement_path.set_data(x + [self.pose[0]],y + [self.pose[1]])

            #Update cop patch
            self.cop_patch.remove()
            self.update_shape(self.pose)
            self.cop_patch = PolygonPatch(self.shape, facecolor=self.default_color, alpha=self.alpha, zorder=2)
            self.ax_list[0].add_patch(self.cop_patch)

            #Update sensor patch
            self.sensor_patch.remove()
            self.sensor_patch = PolygonPatch(self.sensor.shape, facecolor=self.sensor.default_color, alpha=self.sensor.alpha, zorder=2)
            self.ax_list[0].add_patch(self.sensor_patch)


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
            yield self.movement_path,self.scat,self.cop_patch,self.sensor_patch
        
    def animated_exploration(self):
        #<> FIX FRAMES (i.e. stop animation once done)
        self.ani = animation.FuncAnimation(self.fig, self.update, 
            frames=self.num_goals, interval=5, 
            init_func=self.setup_plot, blit=False)
        plt.show()

    
if __name__ == '__main__':
    #Pre-test config
    plt.ion()

    deckard = Cop()
    deckard.animated_exploration()
        
