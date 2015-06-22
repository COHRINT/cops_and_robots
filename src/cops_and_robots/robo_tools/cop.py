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

from pylab import *
import logging

import matplotlib.animation as animation
from matplotlib.colors import cnames

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
    mission_statuses : {'searching', 'capturing', 'done'}
        The possible mission-level statuses of any cop, where:
            * `searching` means the cop is exploring the environment;
            * `capturing` means the cop has detected a robber and is moving 
                to capture it;
            * `done` means all robbers have been captured.

    """
    mission_statuses = ['searching', 'capturing', 'done']

    def __init__(self,
                 name="Deckard",
                 pose=[0, 0.5, 90],
                 fusion_engine_type='particle',
                 planner_type='particle',
                 cop_model='simple',
                 robber_model='random walk'):

        # Superclass and compositional attributes
        super(Cop, self).__init__(name,
                                  pose=pose,
                                  role='cop',
                                  status=['searching', 'without a goal'],
                                  consider_others=True,
                                  color_str='darkgreen')

        # Tracking attributes
        self.found_robbers = {}
        self.planner.use_target_as_goal = False
        self.planner.type = planner_type

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
            1. done (all robots have been captured)
            2. capturing (moving to view pose to capture target)
            3. searching (moving around to gather information)

        """
        detected_robber = any(self.missing_robbers.values()).status[0] \
            == 'detected'

        # <>TODO: Replace with a proper state machine
        if set(self.status[0]) - set(['capturing', 'at goal']) == set([]):
            captured_robber_names = [r.name for r in self.missing_robbers
                                     .values() if r.status[0] == 'detected']
            for robber_name in captured_robber_names:
                self.missing_robbers[robber_name].status[0] = 'captured'
                self.found_robbers = self.missing_robbers.pop(robber_name)
            self.status[0] = 'searching'
        elif len(self.missing_robbers) is 0:
            self.status[0] = 'done'
        elif detected_robber:
            self.status[0] = 'capturing'
        elif self.status[0] != 'capturing':
            self.status[0] = 'searching'

        if self.status[1] is 'stuck':
            logging.warn('{} is {} and {} (moved {}m in last {} time steps).'
              .format(self.name, self.status[0], self.status[1],
                      self.distance_travelled, self.check_last_n))
        elif self.prev_status != self.status:
            logging.info('{} is {} and {}.'.format(self.name, self.status[0],
                                                   self.status[1]))
        self.prev_status = self.status[:]

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
