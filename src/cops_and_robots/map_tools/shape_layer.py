#!/usr/bin/env python
"""Provides a grouping of all map objects.

The element layer accounts for all map objects, grouping them both as a
dictionary of individual objects as well as an accumulation of all
objects in one.

"""
__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

from shapely.geometry import MultiPolygon, Polygon

from cops_and_robots.map_tools.layer import Layer


class ShapeLayer(Layer):
    """A layer object containing all map objects.

    .. image:: img/classes_element_Layer.png

    Parameters
    ----------
    elements
        Dictionary of elements.
    **kwargs
        Arguments for the ``Layer`` superclass.

    """
    def __init__(self, elements, invisible_elements=[], **kwargs):
        super(ShapeLayer, self).__init__(**kwargs)

        self.elements = elements
        self.invisible_elements = invisible_elements
        self.update_static = False

        #<>TODO: plot or not

    def update_plot(self, elements=None, update_static=None, **kwargs):
        """Plot all visible map objects (and their spaces, if visible).

        Parameters
        ----------
        elements : dictionary
            A dictionary with dynamic and static elements

        """

        # <>TODO: Plot Spaces and informations
        if elements is None:
            elements = self.elements

        # Allows update_static to be set elsewhere, but always updates
        # on the first iteration of the animation
        if update_static is None:
            update_static = self.update_static

        self.static_patches = []
        self.dynamic_patches = []
        if update_static:
            for element in self.elements['static']:

                # Re add element
                if element not in self.invisible_elements and element.visible:
                    patch = copy.copy(element.get_patch())
                    self.ax.add_patch(patch)
                    self.static_patches.append(patch)
                # Get element's softmax spaces to plot.
                if element.plot_spaces:
                    pass

        for element in self.elements['dynamic']:
            # Re add element
            if element not in self.invisible_elements and element.visible:
                patch = copy.copy(element.get_patch())
                self.ax.add_patch(patch)
                self.dynamic_patches.append(patch)
            # Get element's softmax spaces to plot.
            if element.plot_spaces:
                pass

        for element in self.elements['information']:
            # Add text and paths
            pass

    def remove(self, remove_static=False):
        if hasattr(self, 'dynamic_patches'):
            for patch in self.dynamic_patches:
                patch.remove()

        if remove_static:
            if hasattr(self, 'static_patches'):
                for patch in self.static_patches:
                    patch.remove()

    def update(self, i=0):
        """Remove dynamic elements and replot.
        """
        # Test stub for the call from __main__
        if __name__ == '__main__':
            for element in self.elements['dynamic']:
                x = np.random.uniform(-5, 5, 1)
                y = np.random.uniform(-5, 5, 1)
                theta = 0
                pose = [x, y, theta]
                element.move_absolute(pose)

        # Allows update_static to be set elsewhere, but always updates
        # on the first iteration of the animation
        self.remove()
        if i == 0:
            self.update_plot(update_static=True)
        else:
            self.update_plot()


def main():
    from cops_and_robots.map_tools.map_elements import MapArea, MapObject

    fig = plt.figure()
    ax = fig.add_subplot(111)

    area1 = MapArea('Area1', [1, 1], pose=[1, 1, 0], visible=True, color_str='blue')
    area2 = MapArea('Area2', [1, 1], pose=[-1, 1, 0], visible=True, color_str='lightblue')
    object1 = MapObject('Object1', [1, 1], pose=[0, 1, 0], visible=True, color_str='red')
    object2 = MapObject('Object2', [1, 1], pose=[0, -1, 0], visible=True, color_str='magenta')

    static_elements = [area1, object1]
    dynamic_elements = [area2, object2]
    information_elements = []
    elements = {'static': static_elements, 'dynamic': dynamic_elements, 'information':information_elements}
    invisible_elements = [area2, object1]

    bounds = [-5, -5, 5, 5]
    sl = ShapeLayer(elements, bounds=bounds)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    # sl = ShapeLayer(elements, invisible_elements=invisible_elements, bounds=[-5, -5, 5, 5])

    ani = animation.FuncAnimation(sl.fig, sl.update,
        frames=xrange(100),
        interval=100,
        repeat=True,
        blit=False)

    plt.show()


if __name__ == '__main__':
    main()
