#!/usr/bin/env python
"""Runs a basic cops_and_robots simulation.

Required Knowledge:
    This module needs to know about the following other modules in the
    cops_and_robots parent module:
        1. ``cop`` to run the simulation from the point of view of a cop.
"""
import logging

from cops_and_robots.robo_tools.robot import Robot
from cops_and_robots.robo_tools.cop import Cop


def main():
    # <>TODO: set up sublime text build system so that this file always runs

    # Set up logger
    # <>TODO: add function name, properly formatted
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.INFO,)

    # Pre-test config
    robber_model = 'stationary'
    deckard = Cop(robber_model=robber_model)
    deckard.map.combined_only = False

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
