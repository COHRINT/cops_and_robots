#!/usr/bin/env python
"""Run a basic cops_and_robots simulation.

"""
import logging

import numpy as np
from cops_and_robots.robo_tools.robot import Robot
from cops_and_robots.robo_tools.cop import Cop


def main():
    # <>TODO: add function name, properly formatted, to logger
    # Set up logger
    logging.basicConfig(format='[%(levelname)-7s] %(funcName)-30s %(message)s',
                        level=logging.INFO,
                        )
    np.set_printoptions(precision=2, suppress=True)

    # Pre-test config
    robber_model = 'static'
    deckard = Cop(robber_model=robber_model, pose_source='python',
                  publish_to_ROS=False)
    deckard.map.combined_only = True

    # Describe simulation
    robber_names = [name for name, role in Robot.all_robots.iteritems()
                    if role == 'robber']
    if len(robber_names) > 1:
        str_ = ', '.join(robber_names[:-1]) + ' and ' + str(robber_names[-1])
    else:
        str_ = robber_names
    logging.info('Simulation started with {} chasing {}.'
                 .format(deckard.name, str_))

    deckard.animated_exploration()

if __name__ == '__main__':
    main()
