#!/usr/bin/env python
"""Run the Cops and Robots simulation.

Configurations can be specified by files in the 'configs' folder.
"""
import logging
import time
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from cops_and_robots.helpers.config import load_config
from cops_and_robots.helpers.storage import Storage
from cops_and_robots.robo_tools.cop import Cop
from cops_and_robots.robo_tools.robber import Robber
from cops_and_robots.robo_tools.robot import Distractor
# <>TODO: @MATT @NICK Look into adding PyPubSub


class CopsAndRobbers(object):
    """Cops and Robbers experimental simulation engine.

    long description of CopsAndRobbers
    
    Parameters
    ----------
    param : param_type, optional
        param_description

    Attributes
    ----------
    attr : attr_type
        attr_description

    Methods
    ----------
    attr : attr_type
        attr_description

    """

    def __init__(self, config_file='config.yaml'):
        # Load configuration files
        self.cfg = load_config(config_file)

       # Configure Python's logging
        logger_level = logging.getLevelName(self.cfg['main']['logging_level'])
        logger_format = '[%(levelname)-7s] %(funcName)-30s %(message)s'
        try:
            logging.getLogger().setLevel(logger_level)
            logging.getLogger().handlers[0]\
                .setFormatter(logging.Formatter(logger_format))
        except IndexError:
            logging.basicConfig(format=logger_format,
                                level=logger_level,
                               )
        np.set_printoptions(precision=self.cfg['main']['numpy_print_precision'],
                            suppress=True)

        # Set up a ROS node (if using ROS)
        if self.cfg['main']['use_ROS']:
            import rospy
            rospy.init_node(self.cfg['main']['ROS_node_name'], 
                            log_level=rospy.DEBUG)

            # Link node to Python's logger
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(logger_format))
            logging.getLogger().addHandler(handler)

        # Play pre-recorded rosbag (process is None if no rosbag selected)
        self.play_data(**self.cfg['data_management']['playback'])

        # Create robbers with config params
        other_robot_names = {'robbers': [], 'distractors':[]}
        self.robbers = {}
        i = 0
        for robber, kwargs in self.cfg['robbers'].iteritems():
            if i >= self.cfg['main']['number_of_agents']['robbers']:
                break
            self.robbers[robber] = Robber(robber, **kwargs)
            logging.info('{} added to simulation.'.format(robber))
            i += 1
            other_robot_names['robbers'].append(robber)

        # Create distractors with config params
        self.distractors = {}
        i = 0
        try:
            for distractor, kwargs in self.cfg['distractors'].iteritems():
                if i >= self.cfg['main']['number_of_agents']['distractors']:
                    break
                self.distractors[distractor] = Distractor(distractor, **kwargs)
                i += 1
                other_robot_names['distractors'].append(distractor)
        except AttributeError:
            logging.debug('No distractors available!')

        # Create cop objects with config params
        self.cops = {}
        i = 0
        for cop, kwargs in self.cfg['cops'].iteritems():
            if i >= self.cfg['main']['number_of_agents']['cops']:
                break
            self.cops[cop] = Cop(cop, other_robot_names=other_robot_names,
                                 rosbag_process=self.rosbag_process, **kwargs)
            logging.info('{} added to simulation.'.format(cop))
            i += 1

        # <>TODO: Replace with message passing
        # <>TODO: Or give as required attribute at cop initialization (check cfg)
        # Give cops references to the robber's actual poses
        for cop in self.cops.values():
            for robber_name, robber in self.robbers.iteritems():
                cop.missing_robbers[robber_name].pose2D = \
                    robber.pose2D
            for distrator_name, distractor in self.distractors.iteritems():
                cop.distracting_robots[distrator_name].pose2D = \
                    distractor.pose2D

        # Give robbers the list of found robots, so they will stop when found
        for robber in self.robbers.values():
            robber.found_robbers = self.cops['Deckard'].found_robbers

        # Describe simulation
        cop_names = self.cops.keys()
        if len(cop_names) > 1:
            cop_str = ', '.join(cop_names[:-1]) + ' and ' + str(cop_names[-1])
        else:
            cop_str = cop_names
        robber_names = self.robbers.keys()
        if len(robber_names) > 1:
            robber_str = ', '.join(robber_names[:-1]) + ' and ' + \
                str(robber_names[-1])
        else:
            robber_str = robber_names
        logging.info('Simulation started with {} chasing {}.'
                     .format(cop_str, robber_str))

        # Set up data logging
        if self.cfg['data_management']['storage']['save_data']:
            self.storage = Storage(**self.cfg['data_management']['storage'])

        # Prepare the map (even if using headless mode)
        self.fig = self.cops['Deckard'].map.fig
        fusion_engine = self.cops['Deckard'].fusion_engine
        self.cops['Deckard'].map.setup_plot(fusion_engine)


        # # STUB TO GENERATE AREA MASKS
        # from cops_and_robots.map_tools.map import find_grid_mask_for_rooms
        # map_ = self.cops['Deckard'].map
        # grid = fusion_engine.filters['Roy'].probability
        # area_masks = find_grid_mask_for_rooms(map_, grid)
        # np.save('coarse_area_masks', area_masks)
        # self.cops['Deckard'].map.plot()

        # return

        # Define timing
        self.start_time = time.time()
        self.max_run_time = self.cfg['main']['max_run_time']
        if self.max_run_time < 0:
            self.max_run_time = 10000000  #<>TODO: not like this

        # Run the simulation
        if self.cfg['main']['headless_mode'] or \
            self.cfg['cops']['Deckard']['map_cfg']['publish_to_ROS']:

            self.headless_mode()
        else:
            self.display_mode()

    def headless_mode(self):
        """Runs the simulation without any animation output.
        """
        i = 0
        while self.cops['Deckard'].mission_planner.mission_status != 'stopped':
            self.update(i)
            i += 1

            # End early
            if i >= self.cfg['main']['max_run_time']:
                break

    def display_mode(self):
        """Animate the exploration of the environment from a cop's perspective.
        """
        ani = animation.FuncAnimation(self.fig,
                                      self.update,
                                      interval=1,
                                      blit=False,
                                      frames=self.max_run_time,
                                      repeat=False,
                                      )
        # <>TODO fix frames (i.e. stop animation once done)
        # <>TODO: break from non-blocking plt.show() gracefully

        # Save the animation
        logging.info(self.cfg['data_management']['animation']['save_animation'])
        if self.cfg['data_management']['animation']['save_animation']:
            folder = '/' + self.cfg['data_management']['animation']['folder']
            filename = self.cfg['data_management']['animation']['filename']
            filename = os.path.dirname(__file__) + folder + filename + '.gif'
            logging.info(filename)
            ani.save(filename, writer='imagemagick', bitrate=1, fps=10);
        plt.show()

    def update(self, i):
        """Update all the major aspects of the simulation and record data.
        """
        logging.debug('Main update frame {}'.format(i))

        # Update all actors
        for cop_name, cop in self.cops.iteritems():
            cop.update(i)

        for robber_name, robber in self.robbers.iteritems():
            robber.update(i)

        for distractor_name, distractor in self.distractors.iteritems():
            distractor.update(i)

        self.cops['Deckard'].map.update(i)

        # Log time
        if self.cfg['main']['log_time']:
            logging.info('Frame {} at time {}.'
                .format(i, time.time() - self.start_time))

        # Store data
        if self.cfg['data_management']['storage']['save_data']:
            d = self.storage.collect_data(self.cops, self.robbers, 
                                          self.distractors)
            last_frame = i + 1 >= self.cfg['main']['max_run_time']

            self.storage.save_frame(i, d, last_frame=last_frame)

        # # Add a new frame to the animation
        # if self.cfg['data_management']['animation']['save_animation']:
        #     animation_folder = self.cfg['data_management']['animation']['animation_folder']
        #     plt.savefig(animation_folder + 'frame_{}.png'.format(i))

        # End the animation if necessary
        max_run_time = self.cfg['main']['max_run_time']
        if i >= max_run_time - 1 and max_run_time > 0:
            plt.close('all')
            logging.info('Simulation ended!')
            if self.cfg['data_management']['storage']['save_data']:
                self.storage.store.close()


    def play_data(self, rosbags={}, folder='data/ACC\ 2016/rosbags/',
                  play_rosbags=True):
        """Plays 1 or more recorded rosbags while the simulation executes.

        Creates a separate process that runs alongside the main Python simulation.
        """
        # End prematurely if not playing data
        if not (play_rosbags and self.cfg['main']['use_ROS']) \
            or len(rosbags)==0:

            self.rosbag_process = None
            return 

        from subprocess import Popen, PIPE, STDOUT

        for _, rosbag in files.iteritems():
            rosbag_file = rosbag_folder + rosbag
            cmd = 'rosbag play -q ' + rosbag_file
            proc = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            logging.info('Created a rosbag process {}!'.format(proc.pid))

        self.rosbag_process = proc

if __name__ == '__main__':
    cnr = CopsAndRobbers()
