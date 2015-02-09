#!/usr/bin/env python
"""Collects all the information a robot has of its environment.

A map has three different parts: its properties, its elements, and its
layers. Properties define key intrinsics of the map (such as its
boundaries). Elements define the collections of physical objects on
the map (robots, walls, etc.). Layers (meant to be toggled for
visibility) represent perspectives on the map: where the map's
elements are, what probability distributions are associated with a
given element, etc.

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``map_obj`` to represent map elements (like walls).
        2. ``shape_layer`` to collect map elements.
        3. ``occupancy_layer`` to represent a grid of occupied space.
        4. ``feasible_layer`` for unoccupied and reachable space.
        5. ``probability_layer`` to represent continuous probability.
        6. ``particle_layer`` to represent discrete probability points.
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

from matplotlib.colors import cnames
import matplotlib.pyplot as plt
from shapely.geometry import Point
from descartes.patch import PolygonPatch

from cops_and_robots.map_tools.map_obj import MapObj
from cops_and_robots.map_tools.shape_layer import ShapeLayer
from cops_and_robots.map_tools.occupancy_layer import OccupancyLayer
from cops_and_robots.map_tools.feasible_layer import FeasibleLayer
from cops_and_robots.map_tools.probability_layer import ProbabilityLayer
from cops_and_robots.map_tools.particle_layer import ParticleLayer
from cops_and_robots.map_tools.human_interface import HumanInterface


class Map(object):
    """Environment map composed of multiple elements and layers.

    :param mapname: the name of the map.
    :type mapname: String.
    :param bounds: map boundaries as [x_min,y_min,x_max,y_max].
    :type bounds: list of floats.
    """

    def __init__(self, mapname, bounds, combined_only=True):
        # Define map properties
        self.mapname = mapname
        self.bounds = bounds  # [x_min,y_min,x_max,y_max] in [m] useful area
        self.outer_bounds = [i * 1.1 for i in self.bounds]
        self.origin = [0, 0]  # in [m]
        self.fig = plt.figure(1, figsize=(12, 10))
        self.particle_color_gain = 400
        self.combined_only = combined_only

        # Define map elements
        self.objects = {}  # For dynamic/static map objects (not robbers/cops)
        self.cops = {}
        self.robbers = {}

        # Define layers
        self.shape_layer = ShapeLayer(bounds=bounds)
        self.occupancy_layer = OccupancyLayer(bounds=bounds)
        self.feasible_layer = FeasibleLayer(bounds=bounds)
        self.particle_layer = {}  # One per robber, plus one combined
        self.probability_layer = {}  # One per robber, plus one combined

    def add_human_sensor(self, human_sensor):
        # Add human sensor for the human interface
        self.human_sensor = human_sensor

    def add_obj(self, map_obj):
        """Append a static ``MapObj`` to the Map

        :param map_obj: MapObj object
        """
        self.objects[map_obj.name] = map_obj
        # self.occupancy_layer.add_obj(map_obj)
        self.shape_layer.add_obj(map_obj)
        self.feasible_layer.define_feasible_regions(self.shape_layer)
        # <>TODO: Update probability layer

    def rem_obj(self, map_obj_name):
        """Remove a ``MapObj`` from the Map by its name

        :param map_obj_name: String
        """
        self.shape_layer.rem_obj(map_obj_name)
        self.feasible_layer.define_feasible_regions(self.shape_layer)
        # self.occupancy_layer.rem_obj(self.objects[map_obj_name])
        del self.objects[map_obj_name]
        # <>TODO: Update probability layer

    def add_cop(self, cop_obj):
        """Add a dynamic ``Robot`` cop from the Map

        :param cop_obj: Map object
        :type cop: MapObj.
        """
        # self.shape_layer.add_obj(cop)
        self.cops[cop_obj.name] = cop_obj

    def rem_cop(self, cop_name):
        """Remove a dynamic ``Robot`` cop from the Map by its name

        :param cop_name: String
        """
        # self.shape_layer.rem_obj(cop_name)
        del self.cops[cop_name]

    def add_robber(self, robber):
        """Add a dynamic ``Robot`` robber from the Map

        :param robber: Robot
        """
        self.robbers[robber.name] = robber
        self.particle_layer[robber.name] = ParticleLayer()
        # <>TODO: Update probability layer

    def rem_robber(self, robber_name):
        """Remove a dynamic ``Robot`` robber from the Map by its name

        :param robber_name: String
        """
        self.shape_layer.rem_obj(robber_name)
        del self.robbers[robber_name]
        del self.probability_layer[robber_name]
        # <>TODO: Update probability layer

    def plot(self,
             robber_name="combined",
             plot_zones=True,
             feasible_region="pose"):
        """Generate one or more probability and occupancy layer

        :param robber_name: String
        """
        # <>TODO: Generate static plot
        pass

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
        arbitrary_particle_layer = next(self.particle_layer.itervalues())
        init_particles = np.zeros((arbitrary_particle_layer.n_particles, 3))
        self.cop_patch = {}
        self.movement_path = {}
        self.camera_patch = {}
        self.robber_patch = {}
        self.particle_scat = {}

        # Set up all robber plots
        if (not self.combined_only) or (len(self.robbers) == 1):
            for robber in self.robbers:
                ax = self.ax_list[robber]
                ax.set_xlim([self.bounds[0], self.bounds[2]])
                ax.set_ylim([self.bounds[1], self.bounds[3]])
                ax.set_title('Tracking {}.'.format(robber))

                # Plot static elements
                self.shape_layer.plot(plot_zones=False, ax=ax)

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

        # Set up combined plot
        if (len(self.ax_list) > 1) or self.combined_only:
            # Plot setup
            self.ax_list['combined'].set_xlim([self.bounds[0], self.bounds[2]])
            self.ax_list['combined'].set_ylim([self.bounds[1], self.bounds[3]])
            self.ax_list['combined'].set_title('Combined tracking of all '
                                               'remaining targets.')

            # Static elements
            self.shape_layer.plot(plot_zones=False,
                                  ax=self.ax_list['combined'])

            # Dynamic elements
            self.cop_patch['combined'] = PolygonPatch(simple_poly)
            self.movement_path['combined'] = movement_path
            self.camera_patch['combined'] = PolygonPatch(simple_poly)

            self.ax_list['combined'].add_patch(self.cop_patch['combined'])
            self.ax_list['combined'].add_line(self.movement_path['combined'])
            self.ax_list['combined'].add_patch(self.camera_patch['combined'])

            self.robber_patch['combined'] = {}
            for robber in self.robbers:
                self.robber_patch['combined'][robber] = \
                    PolygonPatch(simple_poly)
                self.ax_list['combined']\
                    .add_patch(self.robber_patch['combined'][robber])

            init_combined_particles = init_particles.repeat(3, axis=0)
            self.particle_scat['combined'] = self.ax_list['combined']\
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

        # Set up the human interface
        if self.human_sensor:
            HumanInterface(self.fig, self.human_sensor)

        return

    def animation_stream(self):
        """Generate new values for the animation plot, based on an
        updated model of the world.
        """
        while True:
            packet = yield
            for robber_name, pkt in packet.iteritems():
                logging.debug("Updating {}'s animation frame."
                              .format(robber_name))

                (cop_shape, cop_path, camera_shape, robber_shape, particles)\
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

                # Update Particle Filter
                colors = particles[:, 2] * self.particle_color_gain

                self.particle_scat[robber_name].set_array(colors)
                # colors = np.repeat([colors],3,axis=0).T
                # self.particle_scat[robber_name].set_facecolor(colors)
                self.particle_scat[robber_name].set_offsets(particles[:, 0:2])


def set_up_fleming():
    # Make vicon field space object
    net_w = 0.2  # [m] Netting width
    field_w = 10  # [m] field width
    netting = MapObj('Netting', [field_w + net_w, field_w + net_w],
                     has_zones=False)
    field = MapObj('Field', [field_w, field_w], has_zones=False)

    # Make wall objects
    l = 1.2192  # [m] wall length
    w = 0.1524  # [m] wall width
    wall_shape = [l, w]
    poses = ((0, 0, 0),
             (l, l, 0),
             (l * 1.5 + w / 2, l * 1.5 - w / 2, 90),
             (2 * l, 2 * l, 0),
             (l, 3 * l, 0),
             (l * 2.5 + w / 2, l * 1.5 - w / 2, 90),
             (l, 0, 0),
             (l * 1.5 + w / 2, -l * 0.5 + w / 2, 90),
             )

    walls = []
    for i in range(0, len(poses)):
        name = 'Wall ' + str(i)
        pose = poses[i]
        wall = MapObj(name, wall_shape, pose)
        walls.append(wall)

    # Create Fleming map
    bounds = [-field_w / 2, -field_w / 2, field_w / 2, field_w / 2]
    fleming = Map('Fleming', bounds)

    # <>TODO: Fix whatever the hell this was supposed to be
    fleming.add_obj(netting)
    fleming.add_obj(field)
    fleming.rem_obj('Field')
    fleming.rem_obj('Netting')

    # Add walls to map
    for wall in walls:
        fleming.add_obj(wall)

    return fleming

if __name__ == "__main__":
    # fleming = set_up_fleming()

    # fleming.plot('Roy')
    # fleming.probability['Roy'].update(fleming.objects['Wall_4'],'back')
    # fleming.plot('Roy')
    pass
