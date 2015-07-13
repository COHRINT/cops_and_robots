#!/usr/bin/env python
"""Run a basic cops_and_robots simulation.

"""
import logging
import rospy

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

    publish_to_ROS = True

    if publish_to_ROS:
        rospy.init_node('python_node', log_level=rospy.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(levelname)-7s] %(funcName)-30s %(message)s'))
        logging.getLogger().addHandler(handler)

    # Pre-test config
    # <>TODO create a configuration file
    robber_model = 'static'
    deckard = Cop(robber_model=robber_model, pose_source='odom',
                  publish_to_ROS=publish_to_ROS)
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


class ConnectPythonLoggingToROS(logging.Handler):

    MAP = {
        logging.DEBUG:rospy.logdebug,
        logging.INFO:rospy.loginfo,
        logging.WARNING:rospy.logwarn,
        logging.ERROR:rospy.logerr,
        logging.CRITICAL:rospy.logfatal
    }

    def emit(self, record):
        try:
            print('tried')
            self.MAP[record.levelno]("%s: %s" % (record.name, record.msg))
        except KeyError:
            rospy.logerr("unknown log level %s LOG: %s: %s" % (record.levelno, record.name, record.msg))


if __name__ == '__main__':
    main()
