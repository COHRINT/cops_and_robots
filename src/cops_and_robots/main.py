#!/usr/bin/env python
"""Run a basic cops_and_robots simulation.

"""
import logging

import numpy as np
from cops_and_robots.helpers.config import load_config
from cops_and_robots.robo_tools.cop import Cop

def main():
    # Load configuration files
    cfg = load_config()
    main_cfg = cfg['main']
    cop_cfg = cfg['cops']['Deckard']
    publish_to_ROS = cop_cfg['pose_source'] != 'python'

    # Set up logging and printing
    logger_level = logging.getLevelName(main_cfg['logging_level'])
    logger_format = '[%(levelname)-7s] %(funcName)-30s %(message)s'
    logging.basicConfig(format=logger_format,
                        level=logger_level,
                        )
    np.set_printoptions(precision=main_cfg['numpy_print_precision'],
                        suppress=True)

    # Set up a ROS node (if using ROS) and link it to Python's logger
    if publish_to_ROS:
        import rospy
        rospy.init_node('python_node', log_level=rospy.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(logger_format))
        logging.getLogger().addHandler(handler)

    # Create the cop robot with config params
    cop_kwargs = {'robber_model': cop_cfg['robber_model'],
                  'pose_source': cop_cfg['pose_source'],
                  'publish_to_ROS': publish_to_ROS,
                  }
    deckard = Cop(**cop_kwargs)

    # Describe simulation
    robber_names = cfg['robbers'].keys()
    if len(robber_names) > 1:
        str_ = ', '.join(robber_names[:-1]) + ' and ' + str(robber_names[-1])
    else:
        str_ = robber_names
    logging.info('Simulation started with {} chasing {}.'
                 .format(deckard.name, str_))

    deckard.animated_exploration()

if __name__ == '__main__':
    main()
