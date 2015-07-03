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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import cnames
from shapely.geometry import Point

from cops_and_robots.robo_tools.robot import Robot
from cops_and_robots.robo_tools.fusion.fusion_engine import FusionEngine
from cops_and_robots.robo_tools.fusion.camera import Camera
from cops_and_robots.robo_tools.fusion.human import Human


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
    fusion_engine_type : {'discrete','continuous'}
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
                                  mission_status='searching',
                                  consider_others=True,
                                  color_str='darkgreen')

        # Tracking attributes
        self.found_robbers = {}
        self.goal_planner.use_target_as_goal = False
        self.goal_planner.type = goal_planner_type

        # Fusion and sensor attributes
        robber_names = [a.name for a in self.missing_robbers.values()]
        self.fusion_engine = FusionEngine(fusion_engine_type,
                                          robber_names,
                                          self.map.feasible_layer,
                                          self.map.shape_layer,
                                          robber_model)
        self.sensors = {}
        self.sensors['camera'] = Camera((0, 0, 0))
        self.sensors['human'] = Human(self.map, robber_names)
        self.map.add_human_sensor(self.sensors['human'])

        # Animation attributes
        self.update_rate = 1  # [Hz]
        self.show_animation = False
        self.stream = self.map.animation_stream()

        self.prev_status = []

    def update_mission_status(self):
        """Update the cop's high-level mission status.

        Update the cop's status from one of:
            1. retired (all robots have been captured)
            2. searching (moving around to gather information)

        """
        if self.mission_status is 'searching':
            if len(self.missing_robbers) is 0:
                self.mission_status = 'retired'
                self.stop_all_movement()

    def update(self, i=0):
        super(Cop, self).update()

        # Update sensor and fusion information
        # Try to visually spot a robber
        for robber in self.missing_robbers.values():
            if self.sensors['camera'].viewcone.shape.contains(Point(robber.pose2D.pose)):
                robber.mission_status = 'captured'
                logging.info('{} captured!'.format(robber.name))
                self.fusion_engine.filters[robber.name].robber_detected(robber.pose2D.pose)
                self.found_robbers.update({robber.name: self.missing_robbers.pop(robber.name)})
                self.map.rem_robber(robber.name)

        # Update probability model
        self.fusion_engine.update(self.pose2D.pose, self.sensors,
                                  self.missing_robbers)
        # Export the next animation stream
        if self.role == 'cop' and self.show_animation:
            packet = {}
            if not self.map.combined_only:
                for i, robber_name in enumerate(self.missing_robbers):
                    packet[robber_name] = \
                        self._form_animation_packet(robber_name)
            packet['combined'] = self._form_animation_packet('combined')
            return self.stream.send(packet)

    def _form_animation_packet(self, robber_name):
        """Turn all important animation data into a tuple.

        Parameters
        ----------
        robber_name : str
            The name of the robber (or 'combined') associated with this packet.

        Returns
        -------
        tuple
            All important animation parameters.

        """
        # Cop-related values
        cop_shape = self.map_obj.shape
        if len(self.pose_history) < self.goal_planner.stuck_buffer:
            cop_path = np.hsplit(self.pose_history[:, 0:2], 2)
        else:
            cop_path = np.hsplit(self.pose_history[-self.goal_planner.stuck_buffer:, 0:2],
                                 2)

        camera_shape = self.sensors['camera'].viewcone.shape

        # Robber-related values
        particles = self.fusion_engine.filters[robber_name].particles
        if robber_name == 'combined':
            robber_shape = {name: robot.map_obj.shape for name, robot
                            in self.missing_robbers.iteritems()}
        else:
            robber_shape = self.missing_robbers[robber_name].map_obj.shape

        # Form and return packet to be sent
        packet = (cop_shape, cop_path, camera_shape, robber_shape, particles,)
        return packet

    def animated_exploration(self):
        """Start the cop's exploration of the environment, while
        animating the world from the cop's perspective.

        """
        # <>TODO fix frames (i.e. stop animation once done)
        self.show_animation = True
        self.ani = animation.FuncAnimation(self.map.fig,
                                           self.update,
                                           frames=self.num_goals,
                                           interval=5,
                                           init_func=self.map.setup_plot,
                                           blit=False)
        next(self.stream)  # advance the generator once so we can send to it
        plt.show()
