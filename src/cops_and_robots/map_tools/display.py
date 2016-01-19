#!/usr/bin/env python
"""Manages all display elements (figures, axes, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from cops_and_robots.fusion.gaussian_mixture import GaussianMixture

class Display(object):
    """short description of Display

    long description of Display
    
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

    def __init__(self, main_fig, map_, vel_states=None, vel_space=None):
        self.main_fig = main_fig
        self.map = map_

        self.vel_states = vel_states
        if vel_space == None:
            self.vel_space = np.linspace(-3, 3, 100)
        else:
            self.vel_space = vel_space
        if vel_states is not None:
            self._setup_velocity_axes()


    def _setup_velocity_axes(self):
        self.vel_axes = {}
        for name, ax in self.map.axes.iteritems():

            #<>TODO: fix for multiple figures!
            # Create axes with proper bounding boxes
            w = 0.06
            b = ax.get_position()
            x_box = [None] * 4; y_box = [None] * 4  # left, bottom, width, height
            x_box[0] = b.x0
            x_box[1] = b.y1 * 0.868
            x_box[2] = b.width
            x_box[3] = w
            y_box[0] = b.x1
            y_box[1] = b.x0 * 1.743
            y_box[2] = w
            y_box[3] = b.height * 0.705
            vx_ax = self.main_fig.add_axes(x_box)
            vy_ax = self.main_fig.add_axes(y_box)

            # Place important axes
            vx_ax.xaxis.tick_top()
            vx_ax.set_xlabel('x velocity (m/s)')
            vx_ax.xaxis.set_label_position('top') 

            vy_ax.yaxis.tick_right()
            vy_ax.set_ylabel('y velocity (m/s)')
            vy_ax.yaxis.set_label_position('right') 

            # Hide other axes
            vx_ax.get_yaxis().set_visible(False)
            vx_ax.set_axis_bgcolor([0,0,0,0])
            vx_ax.spines['bottom'].set_visible(False)
            vx_ax.spines['left'].set_visible(False)
            vx_ax.spines['right'].set_visible(False)

            vy_ax.get_xaxis().set_visible(False)
            vy_ax.set_axis_bgcolor([0,0,0,0])
            vy_ax.spines['bottom'].set_visible(False)
            vy_ax.spines['left'].set_visible(False)
            vy_ax.spines['top'].set_visible(False)


            self.vel_axes[name] = {'vx': vx_ax,
                                   'vy': vy_ax,
                                   }

    def remove_velocity(self):
        for _, axes in self.vel_axes.iteritems():
            vx_ax = axes['vx']
            vy_ax = axes['vy']

            # Remove lines
            try:
                for line in vx_ax.lines:
                    line.remove()
                for line in vy_ax.lines:
                    line.remove()
            except e:
                logging.error('Exception! {}'.format(e))

            # Remove fill
            try:
                for coll in vx_ax.collections:
                    coll.remove()
                for coll in vy_ax.collections:
                    coll.remove()
            except e:
                logging.error('Exception! {}'.format(e))

    def update_velocity(self):
        self.remove_velocity()
        for name, axes in self.vel_axes.iteritems():
            vx_ax = axes['vx']
            vy_ax = axes['vy']

            v = self.vel_space
            vx = self.vel_states[name].probability.marginal_pdf(x=v, axis=0)
            vy = self.vel_states[name].probability.marginal_pdf(x=v, axis=1)

            # # TEST STUB
            # vx = GaussianMixture(np.random.random(2), (0.5 - np.random.random(2))*5, np.random.random(2))
            # vy = GaussianMixture(np.random.random(2), (0.5 - np.random.random(2))*5, np.random.random(2))

            vx_ax.plot(v, vx, lw=2, color='g')
            vx_ax.fill_between(v, 0, vx, color='g', alpha=0.2)
            vy_ax.plot(vy, v, lw=2, color='g')
            vy_ax.fill_betweenx(v, 0, vy, color='g', alpha=0.2)


    def update(self, i):
        self.map.update(i)
        if self.vel_states is not None:
            self.update_velocity()