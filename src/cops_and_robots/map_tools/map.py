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

from matplotlib.colors import cnames
import matplotlib.pyplot as plt
from shapely.geometry import Point
from descartes.patch import PolygonPatch

from cops_and_robots.helpers.config import load_config
from cops_and_robots.map_tools.map_elements import MapObject, MapArea
from cops_and_robots.map_tools.shape_layer import ShapeLayer
from cops_and_robots.map_tools.occupancy_layer import OccupancyLayer
from cops_and_robots.map_tools.feasible_layer import FeasibleLayer
from cops_and_robots.map_tools.probability_layer import ProbabilityLayer
from cops_and_robots.map_tools.particle_layer import ParticleLayer
from cops_and_robots.map_tools.human_interface import HumanInterface
from cops_and_robots.fusion.gaussian_mixture import GaussianMixture


class Map(object):
    """Environment map composed of multiple elements and layers.

    .. image:: img/classes_Map.png

    Parameters
    ----------
    mapname : str
        The name of the map.
    bounds : array_like, optional
        Map boundaries as [xmin,ymin,xmax,ymax] in [m].
    combined_only : bool, optional
        Whether to show only the combined plot (as opposed to individual plots
        for each robber, plus one combined plot). Defaults to `True`.

    """
    def __init__(self, map_name='fleming', bounds=[-5, -5, 5, 5],
                 plot_robbers=True, map_display_type='particle'):

        # <>TODO: Move to main?
        self.human_cfg = load_config()['human_interface']

        # Define map properties
        self.map_name = map_name
        # <>TODO: Clean this up- add seperate map creation function?
        if self.map_name == 'fleming':
            self.bounds = [-9.5, -3.33, 4, 3.68]
        else:
            self.bounds = bounds  # [x_min,y_min,x_max,y_max] in [m] useful area

        self.plot_robbers = plot_robbers
        self.outer_bounds = [i * 1.1 for i in self.bounds]
        self.origin = [0, 0]  # in [m]
        self.fig = plt.figure(1, figsize=(12, 10))
        # <>TODO: Make display type relative to each robber
        self.display_type = map_display_type

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

    def add_human_sensor(self, human_sensor):
        # Add human sensor for the human interface
        self.human_sensor = human_sensor

    def add_obj(self, map_obj):
        """Append a static ``MapObj`` to the map.

        Parameters
        ----------
        map_obj_name : MapObj
            The object to be added.
        """
        self.objects[map_obj.name] = map_obj
        self.static_elements.append(map_obj)
        self.feasible_layer.define_feasible_regions(self.static_elements)

    def rem_obj(self, map_obj):
        """Remove a ``MapObj`` from the Map.

        map_obj : map_obj
            Map object to remove.
        """
        self.static_elements.remove(map_obj)
        self.feasible_layer.define_feasible_regions(self.static_elements)
        del self.objects[map_obj.name]

    def add_area(self, area):
        self.areas[area.name] = area
        self.static_elements.append(area)

    def rem_area(self, area):
        self.static_elements.remove(area)
        del self.areas[area.name]

    def add_cop(self, cop_obj):
        """Add a dynamic ``Robot`` cop from the Map

        cop_obj : Cop
            The full cop object.
        """
        self.dynamic_elements.append(cop_obj)
        self.cops[cop_obj.name] = cop_obj

    def rem_cop(self, cop_obj):
        """Remove a dynamic ``Robot`` cop

        cop_obj :
            Cop
        """
        self.dynamic_elements.remove(cop_obj)
        del self.cops[cop_obj.name]

    def add_robber(self, robber, filter_):
        # <>TODO: Make generic imaginary robbers
        """Add a dynamic ``Robot`` robber from the Map.

        robber_obj : Robber
            The full robber object.
        """
        if self.plot_robbers is True:
            robber.visible = True
        elif self.plot_robbers is False or robber.name not in self.plot_robbers:
            robber.visible = False

        self.dynamic_elements.append(robber)
        self.robbers[robber.name] = robber

        if self.display_type == 'particle':
            self.add_particle_layer(robber.name, filter_)
        elif self.display_type == 'probability':
            self.add_probability_layer(robber.name, filter_)

    def rem_robber(self, robber):
        """Remove a dynamic ``Robot`` robber.

        robber : obj
            The robber.
        """
        robber.patch.remove()
        self.dynamic_elements.remove(robber)
        del self.robbers[robber.name]
        if self.display_type == 'particle':
            self.rem_particle_layer(robber.name)
        elif self.display_type == 'probability':
            self.rem_probability_layer(robber.name)

    def found_robber(self, robber):
        """Make the robber visible, remove it's particles.

        """
        robber.visible = True
        robber.color = 'darkorange'
        if self.display_type == 'particle':
            self.rem_particle_layer(robber.name)
        elif self.display_type == 'probability':
            self.rem_probability_layer(robber.name)

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
        """Plot the static map.

        """
        # <>TODO: Needs rework
        fig = plt.figure(1, figsize=(12, 10))
        ax = fig.add_subplot(111)
        self.shape_layer.update_plot(update_static=True)

        ax.set_xlim([self.bounds[0], self.bounds[2]])
        ax.set_ylim([self.bounds[1], self.bounds[3]])
        ax.set_title('Experimental environment with landmarks and areas')
        plt.show()

    def setup_plot(self):
        """Create the initial plot for the animation.
        """
        logging.info('Setting up plot')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([self.bounds[0], self.bounds[2]])
        self.ax.set_ylim([self.bounds[1], self.bounds[3]])

        # Set up the human interface
        if self.human_sensor:
            HumanInterface(self.fig, self.human_sensor, **self.human_cfg)

    def update(self, i):
        """
        """
        self.shape_layer.update(i=i)
        if self.display_type == 'particle':
            for particle_layer in self.particle_layers.values():
                particle_layer.update(i=i)
        elif self.display_type == 'probability':
            for probability_layer in self.probability_layers.values():
                probability_layer.update(i=i)


def set_up_fleming(map_):
    """Set up a map as the generic Fleming space configuration.

    """
    # Make vicon field space object
    field_w = 7.5  # [m] field width
    field = MapArea('Field', [field_w, field_w], has_spaces=False)

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

    n_walls = poses.shape[0]
    walls = []
    for i in range(poses.shape[0]):
        name = 'Wall ' + str(i)
        pose = poses[i, :]
        wall = MapObject(name, wall_shape, pose=pose, color_str='sienna',
                         has_spaces=False)
        walls.append(wall)

    # Make rectangular objects (desk, bookcase, etc)
    labels = ['Bookcase', 'Desk', 'Chair', 'Filing Cabinet']
    colors = ['sandybrown', 'sandybrown', 'brown', 'black']
    poses = np.array([[0, -1.2, 270],
                      [-5.5, -2, 00],
                      [3, -2, 180],
                      [-4, -1.3, 270]
                     ])
    sizes = np.array([[0.18, 0.38],
                      [0.61, 0.99],
                      [0.46, 0.41],
                      [0.5, 0.37],
                     ])

    landmarks = []
    for i, pose in enumerate(poses):
        landmark = MapObject(labels[i], sizes[i], pose=pose,
                             color_str=colors[i])
        landmarks.append(landmark)

    # Make odd landmarks
    landmark = MapObject('Filing Cabinet', [0.5, 0.37], pose=[-4, -1.38, 270], color_str='black')
    landmarks.append(landmark)
    # pose = [-9.5, 2.1, 0]
    # shape_pts = Point(pose).buffer(0.2).exterior.coords
    # landmark = MapObject('Frying Pan', shape_pts, pose=pose, has_spaces=False, color_str='slategrey')
    # landmarks.append(landmark)

    # Add walls to map
    for wall in walls:
        map_.add_obj(wall)

    # Add landmarks to map
    for landmark in landmarks:
        map_.add_obj(landmark)

    # Create areas
    labels = ['Study', 'Library', 'Kitchen', 'Billiard Room', 'Hallway',
              'Dining Room']
    colors = ['aquamarine','lightcoral', 'goldenrod', 'sage','cornflowerblue',
              'orchid']
    points = np.array([[[-7.0, -3.33], [-7.0, -1], [-2, -1], [-2, -3.33]],
                       [[-2, -3.33], [-2, -1],[4.0, -1], [4.0, -3.33]],
                       [[-9.5, 1.4], [-9.5, 3.68],[0, 3.68], [0, 1.4]],
                       [[0, 1.4], [0, 3.68],[4, 3.68], [4, 1.4]],
                       [[-9.5, -1], [-9.5, 1.4],[4, 1.4], [4, -1]],
                       [[-9.5, -3.33], [-9.5, -1],[-7, -1], [-7, -3.33]],
                      ])
    for i, pts in enumerate(points):
        centroid = [pts[0,0] + np.abs(pts[2,0] - pts[0,0]) / 2,
                    pts[0,1] + np.abs(pts[1,1] - pts[0,1]) / 2, 0 ]
        area = MapArea(name=labels[i], shape_pts=pts, pose=centroid,
                       color_str=colors[i])
        map_.add_area(area)

    # <>TODO: Include area demarcations
    map_.feasible_layer.define_feasible_regions(map_.static_elements)


if __name__ == '__main__':
    fleming = set_up_fleming()
    fleming.plot()
    fleming.feasible_layer.plot()
