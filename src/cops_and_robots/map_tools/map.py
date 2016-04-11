#!/usr/bin/env python
"""Collects all the information a robot has of its environment.

A map has three different parts: its properties, its elements, and its
layers. Properties define key intrinsics of the map (such as its
boundaries). Elements define the collections of physical objects on
the map (robots, walls, etc.). Layers (meant to be toggled for
visibility) represent perspectives on the map: where the map's
elements are, what probability distributions are associated with a
given element, etc.

"""
from __future__ import division

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"


import logging
import math
import numpy as np

import matplotlib.pyplot as plt
from shapely.geometry import Point

from cops_and_robots.helpers.config import load_config
from cops_and_robots.map_tools.map_elements import MapObject, MapArea
from cops_and_robots.map_tools.shape_layer import ShapeLayer
from cops_and_robots.map_tools.feasible_layer import FeasibleLayer
from cops_and_robots.map_tools.probability_layer import ProbabilityLayer
from cops_and_robots.map_tools.particle_layer import ParticleLayer


class Map(object):
    """Environment map composed of multiple elements and layers.

    .. image:: img/classes_Map.png

    Parameters
    ----------
    mapname : str
        The name of the map.
    bounds : array_like, optional
        Map boundaries as [xmin,ymin,xmax,ymax] in [m].
    plot_robbers : list or bool
        A list of robbers to plot, True for all, False for None.
    map_display_type: {'particle', 'probability'}
        Defines which layer to use.
    combined_only : bool, optional
        Whether to show only the combined plot (as opposed to individual plots
        for each robber, plus one combined plot). Defaults to `True`.
    publish_to_ROS: bool
        Whether to publish an image topic of the map to ROS.
    """
    # TODO: @Config Change plot_robbers to just be a string
    # TODO: @Refactor Seperate map and interface

    def __init__(self, map_name='fleming', bounds=[-5, -5, 5, 5],
                 plot_robbers=True, map_display_type='probability',
                 combined_only=True, publish_to_ROS=False):

        # Define map properties
        # <>TODO: Clean this up- add seperate map creation function?
        self.map_name = map_name
        if self.map_name == 'fleming':
            self.bounds = [-9.5, -3.33, 4, 3.68]
        else:
            self.bounds = bounds  # [x_min,y_min,x_max,y_max] in [m]

        self.plot_robbers = plot_robbers
        self.outer_bounds = [i * 1.1 for i in self.bounds]
        self.origin = [0, 0]  # in [m]

        # <>TODO: Make display type relative to each robber
        self.display_type = map_display_type
        self.combined_only = combined_only

        # Set up ROS elements if using ROS
        self.publish_to_ROS = publish_to_ROS
        if publish_to_ROS:
            from cv_bridge import CvBridge
            import rospy
            from sensor_msgs.msg import Image
            from std_msgs.msg import String
            self.probability_target = 'Roy'
            self.bridge = CvBridge()
            self.image_pub = rospy.Publisher("python_probability_map", Image,
                                             queue_size=10)
            rospy.Subscriber("robot_probability_map", String,
                             self.change_published_ax)

        # Define map elements
        self.objects = {}  # For dynamic/static map objects (not robbers/cops)
        self.areas = {}
        self.cops = {}
        self.robbers = {}

        self.dynamic_elements = []
        self.static_elements = []
        self.information_elements = []
        self.element_dict = {'dynamic': self.dynamic_elements,
                             'static': self.static_elements,
                             'information': self.information_elements}

        # Define layers
        self.shape_layer = ShapeLayer(self.element_dict, bounds=self.bounds)
        self.feasible_layer = FeasibleLayer(bounds=self.bounds)
        self.particle_layers = {}  # One per robber, plus one combined
        self.probability_layers = {}  # One per robber, plus one combined

        # Set up map
        if self.map_name == 'fleming':
            set_up_fleming(self)  # <>TODO: make a generic 'setup map' function
        else:
            pass

    def add_obj(self, map_obj):
        self.objects[map_obj.name] = map_obj
        self.static_elements.append(map_obj)
        self.feasible_layer.define_feasible_regions(self.static_elements)

    def rem_obj(self, map_obj):
        self.static_elements.remove(map_obj)
        self.feasible_layer.define_feasible_regions(self.static_elements)
        del self.objects[map_obj.name]

    def add_robot(self, map_obj):
        # <>TODO: Modify so it can have relations
        self.dynamic_elements.append(map_obj)

    def rem_robot(self, map_obj):
        self.dynamic_elements.remove(map_obj)

    def add_area(self, area):
        self.areas[area.name] = area
        self.static_elements.append(area)

    def rem_area(self, area):
        self.static_elements.remove(area)
        del self.areas[area.name]

    def add_cop(self, cop_obj):
        self.dynamic_elements.append(cop_obj)
        self.cops[cop_obj.name] = cop_obj

    def rem_cop(self, cop_obj):
        self.dynamic_elements.remove(cop_obj)
        del self.cops[cop_obj.name]

    def add_robber(self, robber):
        # <>TODO: Make generic imaginary robbers
        if self.plot_robbers is True:
            robber.visible = True
        elif self.plot_robbers is False:
            robber.visible = False
        elif robber.name not in self.plot_robbers:
            robber.visible = False

        self.dynamic_elements.append(robber)
        self.robbers[robber.name] = robber

    def rem_robber(self, robber):
        robber.patch.remove()
        self.dynamic_elements.remove(robber)
        try:
            if self.fusion_engine is not None:
                if self.display_type == 'particle':
                    self.rem_particle_layer(robber.name)
                elif self.display_type == 'probability':
                    self.rem_probability_layer(robber.name)
        except:
            # <>TODO: actually catch other exceptions here
            logging.debug('No layer to remove.')

        del self.robbers[robber.name]

    def found_robber(self, robber):
        robber.visible = True
        robber.color = 'darkorange'
        try:
            if self.display_type == 'particle':
                self.rem_particle_layer(robber.name)
            elif self.display_type == 'probability':
                self.rem_probability_layer(robber.name)
        except:
            # <>TODO: actually catch other exceptions here
            logging.debug('No layer to remove.')

    def add_particle_layer(self, name, filter_):
        self.particle_layers[name] = ParticleLayer(filter_)

    def rem_particle_layer(self, name):
        self.particle_layers[name].remove()
        del self.particle_layers[name]

    def add_probability_layer(self, name, filter_):
        self.probability_layers[name] = ProbabilityLayer(filter_,
                                                         fig=self.fig,
                                                         bounds=self.bounds)

    def rem_probability_layer(self, name):
        self.probability_layers[name].remove()
        del self.probability_layers[name]

    def plot(self):
        # <>TODO: Needs rework
        fig = plt.figure(1, figsize=(12, 10))
        ax = fig.add_subplot(111)
        self.shape_layer.update_plot(update_static=True)

        for area in self.areas.values():
            ax.add_patch(area.get_patch())

        ax.axis('scaled')
        ax.set_xlim([self.bounds[0], self.bounds[2]])
        ax.set_ylim([self.bounds[1], self.bounds[3]])
        ax.set_title('Experimental environment with landmarks and areas')
        ax.annotate('Kitchen', [-5, 2.5], weight='bold')
        ax.annotate('Billiard Room', [1.5, 2], weight='bold')
        ax.annotate('Library', [0.75, -2.25], weight='bold')
        ax.annotate('Study', [-4.5, -2.25], weight='bold')
        ax.annotate('Dining Room', [-8.75, -1.4], weight='bold')
        ax.annotate('Hallway', [-3.5, 0], weight='bold')
        plt.show()

    def setup_plot(self, fig=None, fusion_engine=None):
        """Create the initial plot for the animation.
        """
        logging.info('Setting up plot')

        if fig is None:
            if plt.get_fignums():
                self.fig = plt.gcf()
            else:
                self.fig = plt.figure(figsize=(14, 10))
        else:
            self.fig = fig

        self.fusion_engine = fusion_engine
        self._setup_axes()
        self._setup_layers()

    def _setup_axes(self):
        self.axes = {}
        if len(self.robbers) == 1:
            name = self.robbers.iterkeys().next()
            self.axes[name] = self.fig.add_subplot(111)
            pos = self.axes[name].get_position()
            print pos
            pos = [pos.x0, pos.y0 * 1.2, pos.width, pos.height]
            print pos
            self.axes[name].set_position(pos)
        elif self.combined_only:
            self.axes['combined'] = self.fig.add_subplot(111)
        else:
            num_axes = len(self.robbers) + 1
            num_rows = int(math.ceil(num_axes / 2))

            i = 0
            for robber_name in self.robbers:
                ax = plt.subplot2grid((num_rows, 4),
                                      (int(math.floor(i / 2)), (i % 2) * 2),
                                      colspan=2
                                      )
                self.axes[robber_name] = ax
                i += 1

            # Add a plot for the combined estimate
            if (num_axes % 2) == 0:
                ax = plt.subplot2grid((num_rows, 4), (num_rows - 1, 2),
                                      colspan=2)
            else:
                ax = plt.subplot2grid((num_rows, 4), (num_rows - 1, 1),
                                      colspan=2)
            self.axes['combined'] = ax

        # Rescale, setup bounds and title
        for ax_name, ax in self.axes.iteritems():
            ax.axis('scaled')
            ax.set_xlim([self.bounds[0], self.bounds[2]])
            ax.set_xlabel('x position (m)')
            ax.set_ylim([self.bounds[1], self.bounds[3]])
            ax.set_ylabel('y position (m)')
            if ax_name == 'combined':
                t = ax.set_title('Combined perception of all robots')
            else:
                t = ax.set_title("Map of {}'s perceived location"
                                 .format(ax_name))

            try:
                if self.fusion_engine.vel_states is not None:
                    t.set_y(1.2)
            except AttributeError:
                logging.debug('No vel states available.')
        # plt.tight_layout()

    def _setup_layers(self):
        # Set up basic layers
        self.shape_layers = {}
        self.feasible_layers = {}
        for ax_name, ax in self.axes.iteritems():
            self.shape_layers[ax_name] = ShapeLayer(self.element_dict,
                                                    bounds=self.bounds,
                                                    ax=ax)

            # Set up probability/particle layers
            if self.fusion_engine is not None:
                filter_ = self.fusion_engine.filters[ax_name]

                self.probability_layers[ax_name] = \
                    ProbabilityLayer(filter_, fig=self.fig, ax=ax, 
                                     bounds=self.bounds)

    def change_published_ax(self, msg):
        self.probability_target = msg.data

    def update(self, i=0):
        # self.shape_layer.update(i=i)
        for ax_name, ax in self.axes.iteritems():
            try:
                self.shape_layers[ax_name].update(i=i)
                # Update probability/particle layers
                if self.fusion_engine is not None:
                    if self.display_type == 'particle':
                        self.particle_layers[ax_name].update(i=i)
                    elif self.display_type == 'probability':
                        self.probability_layers[ax_name].update(i=i)
            except KeyError:
                logging.debug('Robber already removed.')

            if self.publish_to_ROS and ax_name == self.probability_target and \
               i % 1 == 0:
                import cv2
                from cv_bridge import CvBridgeError

                extent = ax.get_window_extent().transformed(
                    self.fig.dpi_scale_trans.inverted())
                self.fig.savefig(ax_name + '.png',
                                 bbox_inches=extent.expanded(1.1, 1.2))
                img = cv2.imread(ax_name + '.png', 1)
                try:
                    self.image_pub.publish(
                        self.bridge.cv2_to_imgmsg(img, "bgr8"))
                except CvBridgeError, e:
                    print e



def set_up_fleming(map_):
    """Set up a map as the generic Fleming space configuration.

    Parameters
    ----------
    map_ : Map 
        The map to add elements.

    """
    # Make vicon field space object
    # <>TODO: Remove field object?
    field_w = 7.5  # [m] field width
    field = MapArea('Field', [field_w, field_w], has_relations=False)

    # Make wall objects
    l = 1.15  # [m] wall length
    w = 0.1524  # [m] wall width
    wall_shape = [l, w]

    poses = np.array([[-7, -1.55, 1],
                      [-7, -1.55 - l, 1],
                      [-7 + l/2 + w/2, -1, 0],
                      [-7 + 3*l/2 + w/2, -1, 0],
                      [-7 + 5*l/2 + w/2, -1, 0],
                      [-2, -1.55, 1],
                      [-2 + 1*l/2 + w/2, -1, 0],
                      [-2 + 3*l/2 + w/2, -1, 0],
                      [-2 + 5*l/2 +w/2, -1, 0],
                      [-7.45 + 1*l/2 + w/2, 1.4, 0],
                      [-7.45 + 3*l/2 + w/2, 1.4, 0],
                      [-7.45 + 5*l/2 + w/2, 1.4, 0],
                      [-7.45 + 7*l/2 + w/2, 1.4, 0],
                      [-7.45 + 9*l/2 + w/2, 1.4, 0],
                      [l/2 + w/2, 1.4, 0],
                      [3*l/2 + w/2, 1.4, 0],
                      [0, 1.4 + l/2, 1],
                      [0, 1.4 + 3*l/2, 1],
                      ])

    poses = poses * np.array([1, 1, 90])

    walls = []
    for i in range(poses.shape[0]):
        name = 'Wall ' + str(i)
        pose = poses[i, :]
        wall = MapObject(name, wall_shape, pose=pose, color_str='sienna',
                         has_relations=False, map_bounds=map_.bounds)
        walls.append(wall)

    # Make rectangular objects (desk, bookcase, etc)
    labels = ['Bookcase', 'Desk', 'Chair', 'Filing Cabinet',
              'Dining Table', 'Mars Poster', 'Cassini Poster',
              'Fridge', 'Checkers Table', 'Fern']
    colors = ['sandybrown', 'sandybrown', 'brown', 'black',
              'brown', 'bisque', 'black',
              'black', 'sandybrown', 'sage']
    poses = np.array([[0, -1.2, 270],  # Bookcase
                      [-5.5, -2, 0],  # Desk
                      [3, -2, 270],  # Chair
                      [-4, -1.32, 270],  # Filing Cabinet
                      [-8.24, -2.15, 270],  # Dining Table
                      [-4.38, 3.67, 270],  # Mars Poster
                      [1.38, 3.67, 270],  # Cassini Poster
                      [-9.1, 3.3, 315],  # Fridge
                      [2.04, 2.66, 270],  # Checkers Table
                      [-2.475, 1.06, 270],  # Fern
                      ])
    sizes = np.array([[0.18, 0.38],  # Bookcase
                      [0.61, 0.99],  # Desk
                      [0.46, 0.41],  # Chair
                      [0.5, 0.37],  # Filing Cabinet
                      [1.17, 0.69],  # Dining Table
                      [0.05, 0.84],  # Mars Poster
                      [0.05, 0.56],  # Cassini Poster
                      [0.46, 0.46],  # Fridge
                      [0.5, 0.5],  # Checkers Table
                      [0.5, 0.5],  # Fern
                      ])

    landmarks = []
    for i, pose in enumerate(poses):
        landmark = MapObject(labels[i], sizes[i], pose=pose,
                             color_str=colors[i], map_bounds=map_.bounds)
        landmarks.append(landmark)

    # Add walls to map
    for wall in walls:
        map_.add_obj(wall)

    # Add landmarks to map
    for landmark in landmarks:
        map_.add_obj(landmark)

    # Create areas
    labels = ['Study', 'Library', 'Kitchen', 'Billiard Room', 'Hallway',
              'Dining Room']
    colors = ['aquamarine', 'lightcoral', 'goldenrod', 'sage',
              'cornflowerblue', 'orchid']
    # points = np.array([[[-7.0, -3.33], [-7.0, -1], [-2, -1], [-2, -3.33]],
    #                    [[-2, -3.33], [-2, -1], [4.0, -1], [4.0, -3.33]],
    #                    [[-9.5, 1.4], [-9.5, 3.68], [0, 3.68], [0, 1.4]],
    #                    [[0, 1.4], [0, 3.68], [4, 3.68], [4, 1.4]],
    #                    [[-9.5, -1], [-9.5, 1.4], [4, 1.4], [4, -1]],
    #                    [[-9.5, -3.33], [-9.5, -1], [-7, -1], [-7, -3.33]],
    #                    ])
    s = 0.0  # coarse scale factor 
    points = np.array([[[-7.0 - s, -3.33], [-7.0 - s, -1 + s], [-2 + s, -1 + s], [-2 + s, -3.33]],
                       [[-2 - s, -3.33], [-2 - s, -1 + s], [4.0, -1 + s], [4.0, -3.33]],
                       [[-9.5, 1.4 - s], [-9.5, 3.68], [0 + s, 3.68], [0 + s, 1.4 - s]],
                       [[0 - s, 1.4 - s], [0 - s, 3.68], [4, 3.68], [4, 1.4 - s]],
                       [[-9.5, -1 - s], [-9.5, 1.4 + s], [4, 1.4 + s], [4, -1 - s]],
                       [[-9.5, -3.33], [-9.5, -1 + s], [-7 + s, -1 + s], [-7 + s, -3.33]],
                       ])


    for i, pts in enumerate(points):
        centroid = [pts[0, 0] + np.abs(pts[2, 0] - pts[0, 0]) / 2,
                    pts[0, 1] + np.abs(pts[1, 1] - pts[0, 1]) / 2, 0]
        area = MapArea(name=labels[i], shape_pts=pts, pose=centroid,
                       color_str=colors[i], map_bounds=map_.bounds)
        map_.add_area(area)

        # Relate landmarks and areas
        for landmark in landmarks:
            if area.shape.intersects(Point(landmark.pose)):
                area.contained_objects[landmark.name] = landmark
                landmark.container_area = area
                landmark.define_relations(map_.bounds)
        area.define_relations(map_.bounds)

    # <>TODO: Include area demarcations
    map_.feasible_layer.define_feasible_regions(map_.static_elements)

def find_grid_mask_for_rooms(map_, grid):
    """Define boolean arrays of which grid cells each area contains.
    """
    area_masks = {}
    pos = grid.pos
    for area_name, area in map_.areas.iteritems():
        area_mask = np.ones_like(grid.prob.flatten())

        for i, pt in enumerate(pos):
            if not area.shape.intersects(Point(pt)):
                area_mask[i] = 0
        area_masks[area_name] = area_mask

    return area_masks

    # # STUB TO GENERATE AREA MASKS - ADD TO main.py
    # from cops_and_robots.map_tools.map import find_grid_mask_for_rooms
    # map_ = self.cops['Deckard'].map
    # grid = fusion_engine.filters['Roy'].probability
    # area_masks = find_grid_mask_for_rooms(map_, grid)
    # np.save('coarse_area_masks', area_masks)
    # self.cops['Deckard'].map.plot()
    # return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fleming = Map()
    fleming.plot()
    # fleming.feasible_layer.plot()
