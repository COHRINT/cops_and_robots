#!/usr/bin/env python
"""Generic definition of a robot. 

Currently subclasses from the iRobotCreate class to allow for 
physical control of an iRobot Create base (if the Robot class is 
configured to control hardware) but could be subclassed to use other
physical hardware in the future. 

A robot has a planner that allows it to select goals and a map to 
keep track of other robots, feasible regions to which it can move, 
an occupancy grid representation of the world, and role-specific 
information (such as a probability layer for the rop robot to keep 
track of where robber robots may be).

Required Knowledge:
    This module and its classes needs to know about the following 
    other modules in the cops_and_robots parent module:
        1. ``iRobot_create`` for hardware control.
        2. ``planner`` to set goal poses and paths.
        3. ``map`` to maintain environment information.
        4. ``map_obj`` to create its own shape in the map.
"""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import math,random
import numpy as np
from matplotlib.colors import cnames

from shapely.geometry import Point

from cops_and_robots.robo_tools.iRobot_create import iRobotCreate
from cops_and_robots.robo_tools.planner import Planner
from cops_and_robots.map_tools.map import Map,set_up_fleming
from cops_and_robots.map_tools.map_obj import MapObj

from time import sleep

class Robot(iRobotCreate):
    """Class definition for the generic robot object. 

    Subclassed by either cops or robbers.

    :param name: the robot's name.
    :type name: String.
    :param pose:
    :type pose:
    :param control_hardware:
    :type control_hardware:
    :param default_color:
    :type default_color:
    :param pose:
    :type pose:
    """

    # all_robot_names=['Deckard','Leon','Pris','Roy','Zhora']
    # all_robot_roles=['cop','robber','robber','robber','robber']
        
    all_robot_names=['Deckard','Roy',]
    all_robot_roles=['cop','robber']

    def __init__(self,name,
                 pose=[0,0.5,0],
                 map_name='fleming',
                 role='robber',
                 status='on the run',
                 planner_type='simple',
                 consider_others=False,
                 control_hardware=False,
                 **kwargs):

        #Figure out which robots to consider
        name = name.capitalize()
        all_robot_names = Robot.all_robot_names
        all_robot_roles = Robot.all_robot_roles

        if consider_others:
            self.other_robot_names = list(set(all_robot_names) - set([name]))
            self.other_robot_roles = all_robot_roles[:]
            del self.other_robot_roles[all_robot_names.index(name)]
        else:
            self.other_robot_names = []
            self.other_robot_roles = []

        if name not in all_robot_names:
            raise ValueError('{} is not a robot name from the list '\
                             'of acceptable robots: {}'\
                             .format(name,all_robot_names))

        #Initialize iRobot Create superclass only if actually controlling hardware
        self.control_hardware = control_hardware
        if self.control_hardware:
            super(Robot,self).__init__()

        #Class attributes
        self.name = name
        self.pose = pose
        self.goal_pose = self.pose[:]
        self.role = role
        self.status = status
        if not map_name: 
            self.map = None
        else:
            self.map = set_up_fleming()
        self.planner = Planner(planner_type,self.map.feasible)
        

        #Movement attributes
        self.move_distance = 0.2 #[m] per time step
        self.rotate_distance = 15 #[deg] per time step
        self.pose_history = np.array(([0,0,0],self.pose)) #list of [x,y,theta] in [m]
        self.check_last_n = 50 #number of poses to look at before assuming stuck
        self.stuck_distance = 0.1 #[m] distance traveled in past self.check_last_n to assume stuck
        self.distance_allowance = 0.1 #[m] how close you can be to a goal
        self.rotation_allowance = 0.5 #[deg] how close you can be to a goal
        self.num_goals = None #number of goals to reach. None for infinite

        #Define MapObj
        shape_pts = Point(pose[0:2]).buffer(iRobotCreate.DIAMETER/2).exterior.coords
        self.map_obj = MapObj(self.name,shape_pts[:],has_zones=False,**kwargs)
        self.update_shape()

        if self.role is 'cop':
            self.map.add_cop(self.map_obj)
        else:
            self.map.add_robber(self.map_obj)

        self.make_others()

        #Start with a goal and a path
        self.goal_pose = self.planner.find_goal_pose()
        self.path,self.path_theta = self.planner.update_path(self.pose)


    def update_shape(self):
        """Update the robot's map_obj.
        """
        self.map_obj.move_shape((self.pose - self.pose_history[-2:,:]).tolist()[0])

    def make_others(self):
        """Generate robot objects for all other robots.

        Create personal belief (not necessarily true!) of other robots, 
        largely regarding their map positions. Their positions are 
        known to the 'self' robot, but this function will be expanded 
        in the future to include registration between robots: i.e., 
        optional pose and information sharing instead of predetermined 
        sharing.
        """
        self.missing_robbers = {} #all are missing to begin with!
        self.known_cops = {}
        for i,robot_name in enumerate(self.other_robot_names):
            #Randomly place the other robots
            x = random.uniform(self.map.bounds[0],self.map.bounds[2])
            y = random.uniform(self.map.bounds[1],self.map.bounds[3])
            theta = random.uniform(0,359)
            pose = [x,y,theta]

            #Add other robots to the map
            if self.other_robot_roles[i] is 'robber':
                new_robber = robber.Robber(robot_name,pose=pose)
                self.map.add_robber(new_robber.map_obj)
                self.missing_robbers[robot_name] = new_robber
            else:
                new_cop = cop.Cop(robot_name,pose=pose)
                self.map.add_cop(new_cop.map_obj)
                self.known_cops[robot_name] = new_cop


    def translate_towards_goal(self,path=None):
        """Move the robot's x,y positions towards a goal point.

        :param path: a line representing the path to the goal.
        :type path: LineString.
        """        
        if not path:
            path = self.path
        next_point = path.interpolate(self.move_distance)
        self.pose[0:2] = (next_point.x,next_point.y)
        print("{} translated to {}".format(self.name,["{:.2f}".format(a) for a in self.pose]))

    def rotate_to_pose(self,theta=None):       
        """Rotate the robot about its centroid towards a goal angle. 
        
        Note:
            All angles represented in degrees.
        """        
        if not theta:
            theta = self.goal_pose[2]
        #Rotate ccw or cw
        angle_diff = theta - self.pose[2]
        rotate_ccw = (abs(angle_diff)<180) and  theta>self.pose[2] \
                     or (abs(angle_diff)>180) and  theta<self.pose[2]
        if rotate_ccw: #rotate ccw
            next_angle = min(self.rotate_distance,abs(angle_diff))
        else:
            next_angle = -min(self.rotate_distance,abs(angle_diff))
        print('Next angle: {:.2f}'.format(next_angle))
        self.pose[2] = (self.pose[2] + next_angle) % 360
        print("{} rotated to {}".format(self.name,["{:.2f}".format(a) for a in self.pose]))

    def check_if_stuck(self):
        """Check if the robot has not moved significantly. 

        Evaluated overover some n time steps. If the robot has not 
        existed for n time steps, it is assumed to not be stuck.
        """
        if len(self.pose_history) > self.check_last_n:
            distance_travelled = 0
            last_poses = self.pose_history[-self.check_last_n:]
            for i,pose in enumerate(last_poses):
                dist = math.sqrt((last_poses[i-1][0] - pose[0]) ** 2 + \
                                  (last_poses[i-1][1] - pose[1]) ** 2)
                distance_travelled += dist
            print('{} travelled {:.2f}m in last {}'.format(self.name,distance_travelled,self.check_last_n))
            if distance_travelled < self.stuck_distance:
                self.status = 'stuck'
            else:
                self.status = ''
                self.update_status()
        #<>NOTE: may want to refactor this and close_enough to join other statuses

    def on_path(self):
        """Check if the robot is on the right path.
        """
        return (abs(self.pose[2] - self.path_theta) < self.rotation_allowance)

    def near_target(self):
        goal_pt = Point(self.goal_pose[0:2])
        approximation_circle = goal_pt.buffer(self.distance_allowance)
        pose_pt = Point(self.pose[0:2])
        return approximation_circle.contains(pose_pt)

    def check_if_close_enough(self,theta=None):
        near = self.near_target()
        goal_theta = self.goal_pose[2]
        pointing_correctly = (abs(self.pose[2] - self.goal_pose[2]) < \
                             self.rotation_allowance)
        close_enough = near and pointing_correctly
        return close_enough

    def update(self,i=0):
        """Update all primary functionality of the robot

        This includes planning and movement for both cops and robbers,
        as well as sensing and map animations for cops.
        """
        print('{} is {}.'.format(self.name,self.status))
        #If stationary or all done, do absolutely nothing.
        if self.status in ('stationary', 'captured all'):
            return

        #Check if close enough to goal
        close_enough = self.check_if_close_enough()
        if close_enough:
            print('close enough')

        #Check if stuck
        self.check_if_stuck()        

        #Generate new path and goal poses
        if close_enough or (self.status in ('detected target','stuck')):
            self.goal_pose = self.planner.find_goal_pose()
            
        #Check if heading in right direction, and move
        if self.on_path() and not self.near_target():
            self.translate_towards_goal()
        elif self.near_target():
            self.rotate_to_pose(self.goal_pose[2])
        else:
            self.rotate_to_pose(self.path_theta)
        self.path,self.path_theta = self.planner.update_path(self.pose)

        #Update sensor and fusion information, if a cop
        if self.role is 'cop':
            for robber in self.missing_robbers.values():
                self.sensors['camera'].detect_robber(robber)
            self.fusion_engine.update(self.pose,self.sensors,self.missing_robbers)

        #Add to the pose history
        self.pose_history = np.vstack((self.pose_history,self.pose[:]))
        self.update_shape()
        self.update_status()

        #Export the next animation stream
        if self.role is 'cop' and self.show_animation:
            packet = {}
            for i,robber_name in enumerate(self.missing_robbers):
                packet[robber_name] = self.form_animation_update_packet(robber_name)
            return self.stream.send(packet)


    def form_animation_update_packet(self,robber_name):

        #Cop-related values
        cop_shape = self.map_obj.shape
        if len(self.pose_history) < self.check_last_n:
            cop_path = np.hsplit(self.pose_history[:,0:2],2)
        else:
            cop_path = np.hsplit(self.pose_history[-self.check_last_n:,0:2],2)

        camera_shape = self.sensors['camera'].viewcone.shape

        #Robber-related values
        particles = self.fusion_engine.filters[robber_name].particles
        robber_shape = self.missing_robbers[robber_name].map_obj.shape

        #Form and return packet
        packet = (cop_shape,cop_path,camera_shape,particles,robber_shape)
        return packet

#Import statements left to the bottom because of subclass circular dependency
import cops_and_robots.robo_tools.cop as cop
import cops_and_robots.robo_tools.robber as robber
