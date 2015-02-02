#!/usr/bin/env python
"""Provides some common functionality for cop robots.

Much of a cop's functionality is defined by the ``robot`` module, but
this module provides cops with the tools it uses to hunt the robbers,
such as:
* sensors (both human and camera) to collect environment information;
* a fusion_engine (either particle or gaussian mixture) to make sense
  of the environment information;
* animation to display its understanding of the world to the human.

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``robot`` for all basic functionality.
        2. ``camera`` for a visual sensor.
        3. ``human`` for a fleshy sensor.
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

    :param pose: the centroid's pose as [x,y,theta] in [m,m,deg].
    :type pose: a 3-element list of floats.
    :param fusion_engine_type: either 'discrete' or 'continuous'.
    :type fusion_engine_type: String.
    :param cop_model: imagined motion model for cops as one of:
        1. 'stationary'
        2. 'random walk'
        3. 'clockwise'
        4. 'counter-clockwise'
    :type cop_model: string.
    :param robber_model: imagined motion model for robbers as one of:
        1. 'stationary'
        2. 'random walk'
        3. 'clockwise'
        4. 'counter-clockwise'
    :type robber_model: string.
    """

    # Define possible statuses as an ordered list
    mission_statuses = ['searching', 'investigating', 'capturing', 'done']

    def __init__(self,
                 name="Deckard",
                 pose=[0, 0.5, 90],
                 fusion_engine_type='discrete',
                 planner_type='particle',
                 cop_model='simple',
                 robber_model='random walk'):

        # Superclass and compositional attributes
        super(Cop, self).__init__(name,
                                  pose=pose,
                                  role='cop',
                                  status=['searching', 'without a goal'],
                                  consider_others=True,
                                  default_color=cnames['darkgreen'])

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
        self.sensors['human'] = Human(self.map.shape_layer.shapes,
                                      robber_names)
        self.update_rate = 1  # [Hz]

        # Animation attributes
        self.show_animation = False
        self.stream = self.map.animation_stream()

    def update_mission_status(self):
        """Update the cop's status from a priority list.

        Update the cop's status from one of:
            1. done (all robots have been captured)
            2. capturing (moving to view pose to capture target)
            3. searching (moving around to gather information)
        """

        detected_robber = any(self.missing_robbers.values()).status[0] \
            == 'detected'

        # <>TODO: Find a more elegant solution
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

        logging.info('{} is {} and {}.'.format(self.name, self.status[0],
                                               self.status[1]))

    def animated_exploration(self):
        """Start the cop's exploration of the environment, while
        animating the world from the cop's perspective.
        """
        # <>TODO FIX FRAMES (i.e. stop animation once done)
        self.show_animation = True
        self.ani = animation.FuncAnimation(self.map.fig,
                                           self.update,
                                           frames=self.num_goals,
                                           interval=5,
                                           init_func=self.map.setup_plot,
                                           blit=False)
        next(self.stream)  # advance the generator once so we can send to it
        plt.show()
