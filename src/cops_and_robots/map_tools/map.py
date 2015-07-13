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

from pylab import *
import logging
import math
import numpy as np

from matplotlib.colors import cnames
import matplotlib.pyplot as plt
from shapely.geometry import Point
from descartes.patch import PolygonPatch

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
    def __init__(self, mapname, bounds, display_type='particle',
                 combined_only=True):
        # Define map properties
        self.mapname = mapname
        self.bounds = bounds  # [x_min,y_min,x_max,y_max] in [m] useful area
        self.outer_bounds = [i * 1.1 for i in self.bounds]
        self.origin = [0, 0]  # in [m]
        self.fig = plt.figure(1, figsize=(12, 10))
        self.display_type = display_type
        self.combined_only = combined_only

        # Define map elements
        self.objects = {}  # For dynamic/static map objects (not robbers/cops)
        self.areas = {}
        self.cops = {}
        self.robbers = {}

        # Define layers
        self.shape_layer = ShapeLayer(bounds=bounds)
        self.occupancy_layer = OccupancyLayer(bounds=bounds)
        self.feasible_layer = FeasibleLayer(bounds=bounds)
        if self.display_type is 'particle':
            self.particle_layer = {}  # One per robber, plus one combined
        else:
            self.probability_layer = {}  # One per robber, plus one combined

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
        # self.occupancy_layer.add_obj(map_obj)
        self.shape_layer.add_obj(map_obj)
        self.feasible_layer.define_feasible_regions(self.shape_layer)

    def rem_obj(self, map_obj_name):
        """Remove a ``MapObj`` from the Map by its name.

        map_obj_name : str
            Name of the map object.
        """
        self.shape_layer.rem_obj(map_obj_name)
        self.feasible_layer.define_feasible_regions(self.shape_layer)
        # self.occupancy_layer.rem_obj(self.objects[map_obj_name])
        del self.objects[map_obj_name]

    def add_area(self, label, area):
        self.areas[label] = area

    def rem_area(self, label):
        del self.areas[label]

    def add_cop(self, cop_obj):
        """Add a dynamic ``Robot`` cop from the Map

        cop_obj : Cop
            The full cop object.
        """
        # self.shape_layer.add_obj(cop)
        self.cops[cop_obj.name] = cop_obj

    def rem_cop(self, cop_name):
        """Remove a dynamic ``Robot`` cop from the Map by its name.

        cop_name : str
            Name of the cop.
        """
        # self.shape_layer.rem_obj(cop_name)
        del self.cops[cop_name]

    def add_robber(self, robber):
        """Add a dynamic ``Robot`` robber from the Map.

        robber_obj : Robber
            The full robber object.
        """
        self.robbers[robber.name] = robber
        if self.display_type == 'particle':
            self.particle_layer[robber.name] = ParticleLayer(bounds=self.bounds)
        else:
            self.probability_layer[robber.name] = ProbabilityLayer(fig=self.fig, bounds=self.bounds)

    def rem_robber(self, robber_name):
        """Remove a dynamic ``Robot`` robber from the Map by its name.

        robber_name : str
            Name of the robber.
        """
        # self.shape_layer.rem_obj(robber_name)
        del self.robbers[robber_name]
        if self.display_type == 'particle':
            del self.particle_layer[robber_name]
        else:
            del self.probability_layer[robber_name]

    def plot(self, show_areas=False):
        """Plot the static map.

        """
        fig = plt.figure(1, figsize=(12, 10))
        ax = fig.add_subplot(111)
        self.shape_layer.plot(plot_spaces=False, ax=ax)

        if show_areas:
            for _, area in self.areas.iteritems():
                area.plot()

        ax.set_xlim([self.bounds[0], self.bounds[2]])
        ax.set_ylim([self.bounds[1], self.bounds[3]])
        ax.set_title('Experimental environment with landmarks and areas')
        plt.show()

    def setup_plot(self):
        """Create the initial plot for the animation.
        """
        self.ax_list = {}
        if len(self.robbers) == 1:
            self.ax_list[self.robbers] = self.fig.add_subplot(111)
        elif self.combined_only:
            self.ax_list['combined'] = self.fig.add_subplot(111)
        else:
            num_axes = len(self.robbers) + 1
            num_rows = int(math.ceil(num_axes / 2))

            for i, robber in enumerate(self.robbers):
                ax = plt.subplot2grid((num_rows, 4),
                                      (int(math.floor(i / 2)), (i % 2) * 2),
                                      colspan=2
                                      )
                self.ax_list[robber] = ax

            # Add a plot for the combined estimate
            if (num_axes % 2) == 0:
                ax = plt.subplot2grid((num_rows, 4), (num_rows - 1, 2),
                                      colspan=2)
            else:
                ax = plt.subplot2grid((num_rows, 4), (num_rows - 1, 1),
                                      colspan=2)
            self.ax_list['combined'] = ax

        # Define generic plot elements
        movement_path = Line2D((0, 0), (0, 0), linewidth=2, alpha=0.4,
                               color=cnames['green'])
        simple_poly = Point((0, 0)).buffer(0.01)
        if self.display_type == 'particle':
            arbitrary_particle_layer = next(self.particle_layer.itervalues())
            init_particles = np.zeros((arbitrary_particle_layer.n_particles, 3))
            self.particle_scat = {}
        else:
            init_distribution = GaussianMixture(1,[0,0],[[1,0],[0,1]])
            self.prob_plot = {}

        self.cop_patch = {}
        self.movement_path = {}
        self.camera_patch = {}
        self.robber_patch = {}

        # Set up all robber plots
        if (not self.combined_only) or (len(self.robbers) == 1):
            for robber in self.robbers:
                ax = self.ax_list[robber]
                ax.set_xlim([self.bounds[0], self.bounds[2]])
                ax.set_ylim([self.bounds[1], self.bounds[3]])
                ax.set_title('Tracking {}.'.format(robber))

                # Plot static elements
                self.shape_layer.plot(plot_spaces=False, ax=ax)

                # Define cop path
                self.cop_patch[robber] = PolygonPatch(simple_poly)
                ax.add_patch(self.cop_patch[robber])

                # Define cop movement path
                self.movement_path[robber] = movement_path
                ax.add_line(self.movement_path[robber])

                # Define camera patch
                self.camera_patch[robber] = PolygonPatch(simple_poly)
                ax.add_patch(self.camera_patch[robber])

                # Define robber patch
                self.robber_patch[robber] = PolygonPatch(simple_poly)
                ax.add_patch(self.robber_patch[robber])

                if self.display_type == 'particle':
                    # Define particle scatter plot
                    self.particle_scat[robber] = \
                        ax.scatter(init_particles[:, 0],
                                   init_particles[:, 1],
                                   c=init_particles[:, 2],
                                   cmap=arbitrary_particle_layer.cmap,
                                   s=arbitrary_particle_layer.particle_size,
                                   lw=arbitrary_particle_layer.line_weight,
                                   alpha=arbitrary_particle_layer.alpha,
                                   marker='.',
                                   vmin=0,
                                   vmax=1
                                   )
                else:
                    # Define probability contour plot
                    self.probability_layer[robber].ax = ax
                    # self.probability_layer[robber].plot(init_distribution)

        # Set up combined plot
        if (len(self.ax_list) > 1) or self.combined_only:
            ax = self.ax_list['combined']
            # Plot setup
            ax.set_xlim([self.bounds[0], self.bounds[2]])
            ax.set_ylim([self.bounds[1], self.bounds[3]])
            ax.set_title('Combined tracking of all '
                                               'remaining targets.')

            # Static elements
            self.shape_layer.plot(plot_spaces=False,
                                  ax=ax)

            # Dynamic elements
            self.cop_patch['combined'] = PolygonPatch(simple_poly)
            self.movement_path['combined'] = movement_path
            self.camera_patch['combined'] = PolygonPatch(simple_poly)

            ax.add_patch(self.cop_patch['combined'])
            ax.add_line(self.movement_path['combined'])
            ax.add_patch(self.camera_patch['combined'])

            self.robber_patch['combined'] = {}
            for robber in self.robbers:
                self.robber_patch['combined'][robber] = \
                    PolygonPatch(simple_poly)
                ax\
                    .add_patch(self.robber_patch['combined'][robber])

            if self.display_type == 'particle':
                init_combined_particles = init_particles.repeat(3, axis=0)
                self.particle_scat['combined'] = ax\
                    .scatter(init_combined_particles[:, 0],
                             init_combined_particles[:, 1],
                             c=init_combined_particles[:, 2],
                             cmap=arbitrary_particle_layer.cmap,
                             s=arbitrary_particle_layer.particle_size,
                             lw=arbitrary_particle_layer.line_weight,
                             alpha=arbitrary_particle_layer.alpha,
                             marker='.',
                             vmin=0,
                             vmax=1
                             )
            else:
                self.probability_layer['combined'] \
                    = ProbabilityLayer(fig=self.fig, ax=ax, bounds=self.bounds)


        # Set up the human interface
        if self.human_sensor:
            HumanInterface(self.fig, self.human_sensor)

    def animation_stream(self):
        """Update the animated plot.

        This is a generator function that takes in packets of animation
        data and yields nothing.

        """
        while True:
            packet = yield
            for robber_name, pkt in packet.iteritems():
                logging.debug("Updating {}'s animation frame."
                              .format(robber_name))

                (cop_shape, cop_path, camera_shape, robber_shape, particles, distribution)\
                    = pkt

                # Update cop patch
                self.cop_patch[robber_name].remove()
                self.cop_patch[robber_name] = \
                    PolygonPatch(cop_shape,
                                 facecolor=cnames['green'],
                                 alpha=0.9,
                                 zorder=2)
                self.ax_list[robber_name]\
                    .add_patch(self.cop_patch[robber_name])

                # Update movement path
                # <>TODO: Figure out why this doesn't work for multiple plots
                self.movement_path[robber_name].set_data(cop_path)

                # Update sensor patch
                self.camera_patch[robber_name].remove()
                self.camera_patch[robber_name] = \
                    PolygonPatch(camera_shape,
                                 facecolor=cnames['yellow'],
                                 alpha=0.3,
                                 zorder=2)
                self.ax_list[robber_name]\
                    .add_patch(self.camera_patch[robber_name])

                # Update robber patch
                if robber_name == 'combined':
                    for robber in self.robbers:
                        self.robber_patch['combined'][robber].remove()
                        self.robber_patch['combined'][robber] = \
                            PolygonPatch(robber_shape[robber],
                                         facecolor=cnames['orange'],
                                         alpha=0.9,
                                         zorder=2)
                        self.ax_list['combined']\
                            .add_patch(self.robber_patch['combined'][robber])
                else:
                    self.robber_patch[robber_name].remove()
                    self.robber_patch[robber_name] = \
                        PolygonPatch(robber_shape,
                                     facecolor=cnames['orange'],
                                     alpha=0.9,
                                     zorder=2)
                    self.ax_list[robber_name]\
                        .add_patch(self.robber_patch[robber_name])

                # Update Filter
                if particles is not None:
                    colors = particles[:, 0]\
                        * next(self.particle_layer.itervalues()).color_gain

                    self.particle_scat[robber_name].set_array(colors)
                    self.particle_scat[robber_name].set_offsets(particles[:, 1:3])
                else:
                    self.probability_layer[robber_name].distribution = distribution
                    self.probability_layer[robber_name].update()


def set_up_fleming(display_type='particle'):
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
                      [-7, -2.55, 1],
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

    landmarks = []
    """
    # Make landmark billiards
    poses = np.array([[2.2, 1.5, 0],
                      [1.2, 1, 0],
                      [1.2, 2.75, 0]
                     ])
    colors = ['yellow', 'blue', 'red', 'purple', 'orange', 'green', 'brown',
              'black']

    for i, pose in enumerate(poses):
        name = 'Ball ' + str(i)
        shape_pts = Point(pose).buffer(0.075).exterior.coords
        landmark = MapObject(name, shape_pts[:], pose=pose, has_spaces=False,
                             color_str=colors[i])
        landmarks.append(landmark)

    # Make landmark glasses
    poses = np.array([[-4.7, 2.3, 0],
                      [-4.8, 2.45, 0],
                      [-4.5, 2.35, 0]
                     ])

    for i, pose in enumerate(poses):
        name = 'Glass ' + str(i)
        shape_pts = Point(pose).buffer(0.06).exterior.coords
        landmark = MapObject(name, shape_pts[:], pose=pose, has_spaces=False,
                             color_str='grey')
        landmarks.append(landmark)
    """

    # Make rectangular objects (desk, bookcase, etc)
    poses = np.array([[0, -1.2, 270],
                      [-5.5, -2, 00],
                      [3, -2, 180]
                     ])
    colors = ['sandybrown', 'sandybrown', 'brown']
    labels = ['Bookcase', 'Desk', 'Chair']
    sizes = np.array([[0.18, 0.38],
                      [0.61, 0.99],
                      [0.46, 0.41]
                     ])
    
    for i, pose in enumerate(poses):
        landmark = MapObject(labels[i], sizes[i], pose=pose,
                             color_str=colors[i])
        landmarks.append(landmark)

    # Make odd landmarks
    landmark = MapObject('Filing Cabinet', [0.5, 0.37], pose=[-4, -1.3, 270], color_str='black')
    landmarks.append(landmark)
    # pose = [-9.5, 2.1, 0]
    # shape_pts = Point(pose).buffer(0.2).exterior.coords
    # landmark = MapObject('Frying Pan', shape_pts, pose=pose, has_spaces=False, color_str='slategrey')
    # landmarks.append(landmark)
    

    # Create Fleming map
    bounds = [-9.5, -3.33, 4, 3.68]
    fleming = Map('Fleming', bounds)

    # Add walls to map
    for wall in walls:
        fleming.add_obj(wall)

    # Add landmarks to map
    for landmark in landmarks:
        fleming.add_obj(landmark)

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
        fleming.add_area(labels[i], area)

    # <>TODO: Include area demarcations

    return fleming


if __name__ == '__main__':
    fleming = set_up_fleming()
    fleming.plot(show_areas=True)
    fleming.feasible_layer.plot()
