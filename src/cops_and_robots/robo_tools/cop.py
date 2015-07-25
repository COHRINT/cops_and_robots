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

from shapely.geometry import Point

from cops_and_robots.robo_tools.robber import ImaginaryRobber
from cops_and_robots.robo_tools.robot import Robot
from cops_and_robots.robo_tools.iRobot_create import iRobotCreate
from cops_and_robots.robo_tools.planner import MissionPlanner
from cops_and_robots.fusion.fusion_engine import FusionEngine
from cops_and_robots.fusion.camera import Camera
from cops_and_robots.fusion.human import Human
from cops_and_robots.map_tools.map_elements import MapObject


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
    mission_planner_defaults = {}
    goal_planner_defaults = {'type_': 'particle',
                             'use_target_as_goal': False}
    path_planner_defaults = {'type_': 'direct'}

    def __init__(self,
                 name,
                 pose=[0, 0, 90],
                 pose_source='python',
                 missing_robber_names=[],
                 other_cop_names=[],
                 robber_model='static',
                 map_cfg={},
                 mission_planner_cfg={},
                 goal_planner_cfg={},
                 path_planner_cfg={},
                 camera_cfg={},
                 **kwargs):
        # Use class defaults for kwargs not included
        mp_cfg = Cop.mission_planner_defaults.copy()
        mp_cfg.update(mission_planner_cfg)
        gp_cfg = Cop.goal_planner_defaults.copy()
        gp_cfg.update(goal_planner_cfg)
        pp_cfg = Cop.path_planner_defaults.copy()
        pp_cfg.update(path_planner_cfg)

        # Configure fusion and map based on goal planner
        if gp_cfg['type_'] == 'particle':
            fusion_engine_type = 'particle'
            map_display_type = 'particle'
        elif gp_cfg['type_'] == 'MAP':
            fusion_engine_type = 'gauss sum'
            map_display_type = 'probability'
        # TODO: Refrence in yaml instead?
        map_cfg.update({'map_display_type': map_display_type})

        # Superclass and compositional attributes
        super(Cop, self).__init__(name,
                                  pose=pose,
                                  pose_source=pose_source,
                                  create_mission_planner=False,
                                  goal_planner_cfg=gp_cfg,
                                  path_planner_cfg=pp_cfg,
                                  map_cfg=map_cfg,
                                  color_str='darkgreen',)

        # Tracking attributes
        self.other_cop_names = other_cop_names
        self.missing_robber_names = missing_robber_names
        self.found_robbers = {}

        # Create mission planner
        self.mission_planner = CopMissionPlanner(self, **mp_cfg)

        # Fusion and sensor attributes
        # <>TODO: Fusion Engine owned and refrenced from imaginary robber?
        self.fusion_engine = FusionEngine(fusion_engine_type,
                                          self.missing_robber_names,
                                          self.map.feasible_layer,
                                          robber_model)
        self.sensors = {}
        self.sensors['camera'] = Camera((0, 0, 0),
                                        element_dict=self.map.element_dict,
                                        **camera_cfg)
        self.map.dynamic_elements.append(self.sensors['camera'].viewcone)

        # Add self to map
        self.map.add_cop(self.map_obj)

        # Make others
        self.make_others()

        # Add human sensor after robbers have been made
        self.sensors['human'] = Human(self.map)
        self.map.add_human_sensor(self.sensors['human'])

    def make_others(self):
        # <>TODO: Make generic, so each robot has an idea of all others
        # <>TODO: Move to back to Robot
        """Generate robot objects for all other robots.

        Create personal belief (not necessarily true!) of other robots,
        largely regarding their map positions. Their positions are
        known to the 'self' robot, but this function will be expanded
        in the future to include registration between robots: i.e.,
        optional pose and information sharing instead of predetermined
        sharing.

        """

        # Robot  MapObject
        shape_pts = Point([0, 0, 0]).buffer(iRobotCreate.DIAMETER / 2)\
            .exterior.coords

        # <>TODO: Implement imaginary class for more robust models
        self.missing_robbers = {}
        for name in self.missing_robber_names:
            self.missing_robbers[name] = ImaginaryRobber(name)
            # Add robber objects to map
            self.missing_robbers[name].map_obj = MapObject(name,
                                                           shape_pts[:],
                                                           has_spaces=False,
                                                           blocks_camera=False,
                                                           color_str='none')
            # <>TODO: allow no display individually for each robber
            self.map.add_robber(self.missing_robbers[name].map_obj)
            # All will be at 0,0,0 until actually pose is given.
            # init_pose =
            # self.missing_robbers[name].map_obj.move_absolute(init_pose)

    def update(self, i=0):
        super(Cop, self).update(i=i)

        # Update sensor and fusion information
        # irobber - Imaginary robber
        for irobber in self.missing_robbers.values():
            point = Point(irobber.pose2D.pose[0:2])
            # Try to visually spot a robber
            if self.sensors['camera'].viewcone.shape.contains(point):
                self.map.found_robber(irobber.map_obj)
                logging.info('{} captured!'.format(irobber.name))
                self.mission_planner.found_robber(irobber.name)
                self.fusion_engine.filters[irobber.name].robber_detected(irobber.pose2D.pose)
                self.found_robbers.update({irobber.name:
                                           self.missing_robbers.pop(irobber.name)})

            # Update robber's shapes
            else:
                self.missing_robbers[irobber.name].map_obj.move_absolute(irobber.pose2D.pose)
            # except:
            #     logging.warn('{} has no pose, and can\'t be detected'
            #                  .format(irobber.name))

        # Update probability model
        self.fusion_engine.update(self.pose2D.pose, self.sensors,
                                  self.missing_robbers)


class CopMissionPlanner(MissionPlanner):
    """The Cop subclass of the generic MissionPlanner
    """
    mission_statuses = ['searching', 'capturing', 'retired']

    def __init__(self, robot, mission_status='searching', target_order=None):
        if target_order is None:
            target = None
        else:
            target = target_order[0]
        self.target_order = target_order

        super(CopMissionPlanner, self).__init__(robot,
                                                mission_status=mission_status,
                                                target=target)

    def update(self):
        """Update the cop's high-level mission status.

        Update the cop's status from one of:
            1. retired (all robots have been captured)
            2. searching (moving around to gather information)

        """
        if self.mission_status is 'searching':
            if len(self.robot.missing_robbers) is 0:
                self.mission_status = 'retired'
                self.stop_all_movement()

    def found_robber(self, name):
        # If no target order, do default behavior
        if self.target_order is None:
            return
        else:
            try:
                self.target_order.remove(name)
                if self.target_order == []:
                    self.target_order = None
                    logging.info('Completed target order')
                else:
                    self.target = self.target_order[0]
                    logging.info('New target: {}'.format(self.target))
            except:
                logging.info('{} is not in target order'.format(name))
