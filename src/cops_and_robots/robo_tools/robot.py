#!/usr/bin/env python
"""Generic definition of a robot. Subclasses from the iRobotCreate 
    class to allow for physical control of an iRobot Create base if 
    the Robot class is configured to control hardware. Also Subclasses 
    from MapObj to provide a visualizable object on the map.

    A robot has a planner that allows it to select goals and a map to 
    keep track of other robots, feasible regions to which it can move, 
    an occupancy grid representation of the world, and role-specific 
    information (such as a probability layer for the rop robot to keep 
    track of where robber robots may be)."""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import math
import numpy as np
from cops_and_robots.robo_tools.iRobot_create import iRobotCreate
from cops_and_robots.robo_tools.planner import Planner
from cops_and_robots.map_tools.map import Map
from cops_and_robots.map_tools.map_obj import MapObj

class Robot(MapObj):
    """Class for controlling iRobot Create. Will generate a 'base' thread to maintain
    serial communication with the iRobot base.

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
        
    def __init__(self,name,control_hardware=False,planner_type='simple',
                 role='Robber',**kwargs):
        #Superclass attributes
        self.define_shape_pts()
        super(Robot,self).__init__(name,self.shape_pts,**kwargs)
        #Class attributes
        self.control_hardware = control_hardware
        self.planner = Planner(type=planner_type)
        self.map = Map()
        self.role = role

    def define_shape_pts(self):
        """Define a circle with radius ROBOT_DIAMETER/2 around pose.
        """
        a = np.linspace(0,2 * math.pi, iRobotCreate.RESOLUTION)
        circ_x = [(iRobotCreate.DIAMETER / 2 * math.sin(b)) for b in a]
        circ_y = [(iRobotCreate.DIAMETER / 2 * math.cos(b)) for b in a]
        self.shape_pts = zip(circ_x,circ_y)

    def update_shape(self,pose=(0,0,0)):
        """Given the robot's pre-defined shape, update the shape to 
        the robot's current pose.

        :param pose:
        """
        new_obj = MapObj(self.name,self.shape_pts,pose=pose,
                         default_color=self.default_color)
        self.shape = new_obj.shape

