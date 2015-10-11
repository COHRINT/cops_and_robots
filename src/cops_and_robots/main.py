#!/usr/bin/env python
"""Run a basic cops_and_robots simulation.

"""
import logging
import time
import sys
import os

import numpy as np
import pandas as pd
from pandas import HDFStore
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
    storage_cfg = cfg['data_logging']
 
   # Set up logging and printing
    logger_level = logging.getLevelName(main_cfg['logging_level'])
    logger_format = '[%(levelname)-7s] %(funcName)-30s %(message)s'

    try:
        logging.getLogger().setLevel(logger_level)
        logging.getLogger().handlers[0].setFormatter(logging.Formatter(logger_format))
    except IndexError:
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
    if main_cfg['playback_rosbags'] is None:
        rosbag_process = None
    else:
        rosbag_process = playback_mode(main_cfg['playback_rosbags'])

    cops = {}
    for cop, kwargs in cop_cfg.iteritems():
        cops[cop] = Cop(cop, rosbag_process=rosbag_process, **kwargs)
        logging.info('{} added to simulation'.format(cop))

    # Create robbers with config params
    robbers = {}
    for robber, kwargs in robber_cfg.iteritems():
        robbers[robber] = Robber(robber, **kwargs)
        logging.info('{} added to simulation'.format(robber))

    # Create distractors with config params
    distractors = {}
    try:
        for distractor, kwargs in distractor_cfg.iteritems():
            distractors[distractor] = Distractor(distractor, **kwargs)
    except AttributeError:
        logging.debug('No distractors available!')

    # <>TODO: Replace with message passing
    # <>TODO: Or give as required attribute at cop initialization (check with config)
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

    # Set up data logging
    storage = Storage(storage_cfg)

    # Start the simulation
    fig = cops['Deckard'].map.fig
    fusion_engine = cops['Deckard'].fusion_engine
    cops['Deckard'].map.setup_plot(fusion_engine)
    sim_start_time = time.time()


    if cop_cfg['Deckard']['map_cfg']['publish_to_ROS'] or main_cfg['headless_mode']:
        headless_mode(cops, robbers, distractors, main_cfg, sim_start_time, storage)
    else:
        animated_exploration(fig, cops, robbers, distractors, main_cfg, sim_start_time, storage)


def headless_mode(cops, robbers, distractors, main_cfg, sim_start_time, 
                  storage):
    i = 0
    while cops['Deckard'].mission_planner.mission_status != 'stopped':
        update(i, cops, robbers, distractors, main_cfg, sim_start_time, storage)

        i += 1
        # For ending early
        if i >= main_cfg['max run time']:
            break

def playback_mode(rosbags):
    """Plays 1 or more recorded rosbags while the simulation executes.
    """
    # <>TODO: parametrize the inputs
    from subprocess import Popen, PIPE, STDOUT

    rosbag_folder = 'data/ACC\ 2016/rosbags/'
    for _, rosbag in rosbags.iteritems():
        rosbag_file = rosbag_folder + rosbag
        cmd = 'rosbag play -q ' + rosbag_file
        proc = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        logging.info('Created a rosbag process {}!'.format(proc.pid))
    return proc
    # bag = rosbag.Bag(rosbag_file)
    # for topic, msg, t in bag.read_messages(topics=['/human_sensor', '/tf']):
    #     print msg
    # bag.close()

def animated_exploration(fig, cops, robbers, distractors, main_cfg, 
                         sim_start_time, storage):
    """Animate the exploration of the environment from a cop's perspective.

    """
    # <>TODO fix frames (i.e. stop animation once done)
    max_run_time = main_cfg['max run time']
    if max_run_time < 0:
        max_run_time = 10000000  #<>TODO: not like this

    ani = animation.FuncAnimation(fig,
                                  update,
                                  fargs=[cops, robbers, distractors, main_cfg, sim_start_time, storage],
                                  interval=10,
                                  blit=False,
                                  frames=250,#max_run_time,
                                  repeat=False,
                                  )
    # <>TODO: break from non-blocking plt.show() gracefully
    # ani.save('camera_and_feasible_region.gif', writer='imagemagick', fps=10);
    plt.show()


def update(i, cops, robbers, distractors, main_cfg, sim_start_time, storage):
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

    # Save data
    d = {}
    for record, record_value in storage.records.iteritems():
        if 'robot positions' == record:
            if record_value == 'all':
                d['Deckard position'] = cops['Deckard'].pose2D.pose
                d['Roy position'] = robbers['Roy'].pose2D.pose

        if 'grid probability' == record:
            if record_value == '4D':
                all_dims = True
            else:
                all_dims = False
            #<>TODO: assume more than Roy and Deckard
            d['grid probability'] = cops['Deckard'].fusion_engine\
                .filters['Roy'].probability.as_grid(all_dims)
            d['grid probability'] = d['grid probability'].flatten()

    storage.save_frame(i, d)

    max_run_time = main_cfg['max run time']
    if i >= max_run_time - 1 and max_run_time > 0:
        plt.close('all')
        logging.info('Simulation ended!')
    #<>TODO: have general 'save animation' setting
    # plt.savefig('animation/frame_{}.png'.format(i))

class Storage(object):
    """docstring for Storage"""

    def __init__(self, storage_cfg=None, filename='data', use_prefix=True, use_suffix=True):
        self.file_path = directory = os.path.dirname(__file__) + 'data/ICCPS 2016/'
        self.set_filename(filename, use_prefix, use_suffix)

        self.store = HDFStore(self.filename)
        if storage_cfg is not None:
            self.records = storage_cfg['record_data']
        else:
            self.records = {'grid probability':'2D', 
                            'robot positions':'all'
                            }

        self.dfs = {}

    def set_filename(self, filename, use_prefix, use_suffix):
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)

        if use_prefix:
            self.filename_prefix = 'cnr_'
        else:
            self.filename_prefix = ''

        if use_suffix:
            self.filename_suffix = '_trial'
        else:
            self.filename_suffix = ''
        self.filename_extension = '.hd5'

        self.filename = self.file_path + self.filename_prefix + filename \
            + self.filename_suffix + self.filename_extension

        # Check for duplicate files
        i = 0
        while os.path.isfile(self.filename):
            i += 1
            ind = self.filename.find(self.filename_suffix)
            l = len(self.filename_suffix)
            self.filename = self.filename[:ind + l] + '_' + str(i) + self.filename_extension

    def save_frame(self, frame_i, data):
        """Expects data as a dict
        """
        for key, value in data.iteritems():
            key = key.lower().replace(' ', '_')
            try:
                new_df = pd.DataFrame(value, columns=[str(frame_i + 1)])
                self.dfs[key] = self.dfs[key].join(new_df)
            except:
                self.dfs[key] = pd.DataFrame(value)

            self.store.put(key, self.dfs[key])


if __name__ == '__main__':
    main('config_iccps.yaml')



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