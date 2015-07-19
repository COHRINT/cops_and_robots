#!/usr/bin/env python
"""Provides some common functionality for cop robots.

Much of a cop's functionality is defined by the ``robot`` module, but
this module provides cops with the tools it uses to hunt the robbers,
such as:
    * sensors (both human and camera) to collect environment information;
    * a fusion_engine (either particle or gaussian mixture) to make sense
      of the environment information;
    * animation to display its understanding of the world to the human.

"""
__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import logging
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import cnames
from shapely.geometry import Point

import cops_and_robots.robo_tools.robber as robber_module
from cops_and_robots.robo_tools.robot import Robot
from cops_and_robots.fusion.fusion_engine import FusionEngine
from cops_and_robots.fusion.camera import Camera
from cops_and_robots.fusion.human import Human


class Cop(Robot):
    """The Cop subclass of the generic robot type.

    Cops extend the functionality of basic robots, providing sensing (both
    camera-based and human) and a fusion engine.

    .. image:: img/classes_Cop.png

    Parameters
    ----------
    name : str, optional
        The cop's name (defaults to 'Deckard').
    pose : list of float, optional
        The cop's initial [x, y, theta] (defaults to [0, 0.5, 90]).
    fusion_engine_type : {'particle','gauss_sum'}
        For particle filters or gaussian mixture filters, respectively.
    planner_type: {'simple', 'particle', 'MAP'}
        The cop's own type of planner.
    cop_model: {'stationary', 'random walk', 'clockwise', 'counterclockwise'}
        The type of planner this cop believes other cops use.
    robber_model: {'stationary', 'random walk', 'clockwise',
      'counterclockwise'}
        The type of planner this cop believes robbers use.

    Attributes
    ----------
    fusion_engine
    planner
    found_robbers : dict
        All robbers found so far.
    sensors : dict
        All sensors owned by the cop.
    mission_statuses : {'searching', 'capturing', 'retired'}
        The possible mission-level statuses of any cop, where:
            * `searching` means the cop is exploring the environment;
            * `capturing` means the cop has detected a robber and is moving
                to capture it;
            * `retired` means all robbers have been captured.

    """
    mission_statuses = ['searching', 'capturing', 'retired']

    def __init__(self,
                 name="Deckard",
                 pose=[0, 0, 90],
                 pose_source='python',
                 publish_to_ROS=False,
                 fusion_engine_type='particle',
                 goal_planner_type='particle',
                 cop_model='simple',
                 robber_model='random walk'):

        # Superclass and compositional attributes
        super(Cop, self).__init__(name,
                                  pose=pose,
                                  pose_source=pose_source,
                                  publish_to_ROS=publish_to_ROS,
                                  role='cop',
                                  map_display_type=fusion_engine_type,
                                  mission_status='searching',
                                  consider_others=True,
                                  color_str='darkgreen')

        # Tracking attributes
        self.found_robbers = {}
        self.goal_planner.use_target_as_goal = False
        self.goal_planner.type = goal_planner_type

        # Fusion and sensor attributes
        robber_names = [a for a in self.other_robots.keys()]
        self.fusion_engine = FusionEngine(fusion_engine_type,
                                          robber_names,
                                          self.map.feasible_layer,
                                          robber_model)
        self.sensors = {}
        self.sensors['camera'] = Camera((0, 0, 0), element_dict=self.map.element_dict)
        self.map.dynamic_elements.append(self.sensors['camera'].viewcone)

        # Make others
        self.make_others()

        # Add human sensor after robbers have been made
        self.sensors['human'] = Human(self.map)
        self.map.add_human_sensor(self.sensors['human'])

        # Animation attributes
        self.update_rate = 1  # [Hz]
        self.show_animation = False

        self.prev_status = []

    def make_others(self):
        """Generate robot objects for all other robots.

        Create personal belief (not necessarily true!) of other robots,
        largely regarding their map positions. Their positions are
        known to the 'self' robot, but this function will be expanded
        in the future to include registration between robots: i.e.,
        optional pose and information sharing instead of predetermined
        sharing.
        """

        self.missing_robbers = {}  # all are missing to begin with!
        self.known_cops = {}

        for name, role in self.other_robots.iteritems():

            # Randomly place the other robots
            feasible_robber_generated = False
            while not feasible_robber_generated:
                x = random.uniform(self.map.bounds[0], self.map.bounds[2])
                y = random.uniform(self.map.bounds[1], self.map.bounds[3])
                if self.map.feasible_layer.pose_region.contains(Point([x, y])):
                    feasible_robber_generated = True

            theta = random.uniform(0, 359)
            pose = [x, y, theta]

            # Add other robots to the map
            if role == 'robber':
                new_robber = robber_module.Robber(name, pose=pose)
                # <>TODO: Allow Guass
                self.map.add_robber(new_robber.map_obj, self.fusion_engine.filters[name].particles)
                self.missing_robbers[name] = new_robber
            else:
                new_cop = cop_module.Cop(name, pose=pose)
                self.map.add_cop(new_cop.map_obj)
                self.known_cops[name] = new_cop
            logging.debug('{} added'.format(name))

    def update_mission_status(self):
        # <>TODO: Replace with MissionPlanner
        """Update the cop's high-level mission status.

        Update the cop's status from one of:
            1. retired (all robots have been captured)
            2. searching (moving around to gather information)

        """
        if self.mission_status is 'searching':
            if len(self.missing_robbers) is 0:
                self.mission_status = 'retired'
                self.mission_planner.stop_all_movement()

    def update(self, i=0):
        super(Cop, self).update()

        # Update sensor and fusion information
        # Try to visually spot a robber
        for robber in self.missing_robbers.values():
            if self.sensors['camera'].viewcone.shape.contains(Point(robber.pose2D.pose)):
                robber.mission_status = 'captured'
                logging.info('{} captured!'.format(robber.name))
                self.fusion_engine.filters[robber.name].robber_detected()
                # self.map.rem_robber(robber.map_obj)
                self.map.found_robber(robber.map_obj)
                self.found_robbers.update({robber.name: self.missing_robbers.pop(robber.name)})

        # Update probability model
        self.fusion_engine.update(self.pose2D.pose, self.sensors,
                                  self.missing_robbers)

        # print id(self.fusion_engine.filters['Zhora'].particles)
        # if self.fusion_engine.filters['Zhora'].particles == 'Finished':
        #     print 'done'

        # Export the next animation stream
        if self.show_animation:
            self.map.update(i)

    def animated_exploration(self):
        """Start the cop's exploration of the environment, while
        animating the world from the cop's perspective.

        """
        # <>TODO fix frames (i.e. stop animation once done)
        self.show_animation = True
        self.ani = animation.FuncAnimation(self.map.fig,
                                           self.update,
                                           init_func=self.map.setup_plot,
                                           frames=self.num_goals,
                                           interval=5,
                                           blit=False)
        plt.show()
