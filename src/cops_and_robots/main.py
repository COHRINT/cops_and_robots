#!/usr/bin/env python
"""Run a basic cops_and_robots simulation.

"""
import logging

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cops_and_robots.helpers.config import load_config
from cops_and_robots.robo_tools.cop import Cop
from cops_and_robots.robo_tools.robber import Robber
# <>TODO: @MATT @NICK Look into adding PyPubSub


def main():
    # Load configuration files
    cfg = load_config()
    main_cfg = cfg['main']
    cop_cfg = cfg['cops']
    robber_cfg = cfg['robbers']

    # Set up logging and printing
    logger_level = logging.getLevelName(main_cfg['logging_level'])
    logger_format = '[%(levelname)-7s] %(funcName)-30s %(message)s'
    logging.basicConfig(format=logger_format,
                        level=logger_level,
                        )
    np.set_printoptions(precision=main_cfg['numpy_print_precision'],
                        suppress=True)

    # Set up a ROS node (if using ROS) and link it to Python's logger
    if main_cfg['use_ROS']:
        import rospy
        rospy.init_node(main_cfg['ROS_node_name'], log_level=rospy.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(logger_format))
        logging.getLogger().addHandler(handler)

    # Create cops with config params
    robber_names = robber_cfg.keys()
    cops = {}
    for cop, kwargs in cop_cfg.iteritems():
        cops[cop] = Cop(cop, missing_robber_names=robber_names, **kwargs)
        logging.info('{} added to simulation'.format(cop))

    # Create robbers with config params
    robbers = {}
    for robber, kwargs in robber_cfg.iteritems():
        robbers[robber] = Robber(robber, **kwargs)
        logging.info('{} added to simulation'.format(robber))

    # <>TODO: Replace with message passing
    # Give cops references to the robbers actual pose
    for cop in cops.values():
        for robber_name, robber in robbers.iteritems():
            cop.missing_robbers[robber_name].pose = \
                robber.pose2D.pose
    for robber in robbers.values():
        robber.found_robbers = cops['Deckard'].found_robbers

    # Describe simulation
    cop_names = cops.keys()
    if len(cop_names) > 1:
        cop_str = ', '.join(cop_names[:-1]) + ' and ' + str(cop_names[-1])
    else:
        cop_str = cop_names

    robber_names = robbers.keys()
    if len(robber_names) > 1:
        robber_str = ', '.join(robber_names[:-1]) + ' and ' + \
            str(robber_names[-1])
    else:
        robber_str = robber_names

    logging.info('Simulation started with {} chasing {}.'
                 .format(cop_str, robber_str))

    # Create Visualization
    def update(i=0):
        logging.debug('Main update {}'.format(i))

        for cop_name, cop in cops.iteritems():
            cop.update(i)

        for robber_name, robber in robbers.iteritems():
            robber.update(i)

        cops['Deckard'].map.update(i)

    figure = cops['Deckard'].map.fig

    def init():
        cops['Deckard'].map.setup_plot()

    def animated_exploration():
        """Start the cop's exploration of the environment, while
        animating the world from the cop's perspective.

        """
        # <>TODO fix frames (i.e. stop animation once done)
        ani = animation.FuncAnimation(figure,
                                      update,
                                      init_func=init,
                                      interval=5,
                                      blit=False)
        plt.show()

    animated_exploration()


if __name__ == '__main__':
    main()

    # # Example Cop
    # goal_planner_cfg = {'type': {'level_lower': 'simple'}}
    # cop = Cop("Niles", goal_planner_cfg)
