#!/usr/bin/env python
"""Run a basic cops_and_robots simulation.

"""
import logging
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cops_and_robots.helpers.config import load_config
from cops_and_robots.robo_tools.cop import Cop
from cops_and_robots.robo_tools.robber import Robber
from cops_and_robots.robo_tools.robot import Distractor
# <>TODO: @MATT @NICK Look into adding PyPubSub


def main(config_file=None):
    # Load configuration files
    cfg = load_config(config_file)
    main_cfg = cfg['main']
    cop_cfg = cfg['cops']
    robber_cfg = cfg['robbers']
    distractor_cfg = cfg['distractors']

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
    cops = {}
    for cop, kwargs in cop_cfg.iteritems():
        cops[cop] = Cop(cop, **kwargs)
        logging.info('{} added to simulation'.format(cop))

    # Create robbers with config params
    robbers = {}
    for robber, kwargs in robber_cfg.iteritems():
        robbers[robber] = Robber(robber, **kwargs)
        logging.info('{} added to simulation'.format(robber))

    # Create distractors with config params
    distractors = {}
    for distractor, kwargs in distractor_cfg.iteritems():
        distractors[distractor] = Distractor(distractor, **kwargs)

    # <>TODO: Replace with message passing
    # Give cops references to the robber's actual poses
    for cop in cops.values():
        for robber_name, robber in robbers.iteritems():
            cop.missing_robbers[robber_name].pose2D = \
                robber.pose2D
        for distrator_name, distractor in distractors.iteritems():
            cop.distracting_robots[distrator_name].pose2D = \
                distractor.pose2D
    # Give robber the cops list of found robots, so they will stop when found
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

    # Start the simulation
    fig = cops['Deckard'].map.fig
    fusion_engine = cops['Deckard'].fusion_engine
    cops['Deckard'].map.setup_plot(fusion_engine)
    sim_start_time = time.time()

    if cop_cfg['Deckard']['map_cfg']['publish_to_ROS']:
        headless_mode(cops, robbers, distractors, main_cfg, sim_start_time)
    else:
        animated_exploration(fig, cops, robbers, distractors, main_cfg, sim_start_time)


def headless_mode(cops, robbers, distractors, main_cfg, sim_start_time):
    i = 0
    while cops['Deckard'].mission_planner.mission_status != 'stopped':
        update(i, cops, robbers, distractors, main_cfg, sim_start_time)
        i += 1
        # For ending early
        # if i == 101:
        #     break


def animated_exploration(fig, cops, robbers, distractors, main_cfg, sim_start_time):
    """Animate the exploration of the environment from a cop's perspective.

    """
    # <>TODO fix frames (i.e. stop animation once done)
    ani = animation.FuncAnimation(fig,
                                  update,
                                  fargs=[cops, robbers, distractors, main_cfg, sim_start_time],
                                  interval=10,
                                  blit=False,
                                  repeat=False,
                                  )
    # <>TODO: break from non-blocking plt.show() gracefully
    plt.show()


def update(i, cops, robbers, distractors, main_cfg, sim_start_time):
    logging.debug('Main update frame {}'.format(i))

    for cop_name, cop in cops.iteritems():
        cop.update(i)

    for robber_name, robber in robbers.iteritems():
        robber.update(i)

    for distractor_name, distractor in distractors.iteritems():
        distractor.update(i)

    cops['Deckard'].map.update(i)
    if main_cfg['log_time']:
        logging.info('Frame {} at time {}.'.format(i, time.time() - sim_start_time))

    # plt.savefig('animation/frame_{}.png'.format(i))

if __name__ == '__main__':
    main()






# import cv2
# from cv_bridge import CvBridge, CvBridgeError
# import rospy
# from sensor_msgs.msg import Image
# # <>TODO: Lazy init for ros

# self.bridge = CvBridge()
#         self.image_pub = rospy.Publisher("test_prob_layer", Image)
#         rospy.init_node('Probability_Node')


#     plt.savefig('foo.png')
#         img = cv2.imread('foo.png', 1)
#         try:
#             self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "passthrough"))
#         except CvBridgeError, e:
#             print e