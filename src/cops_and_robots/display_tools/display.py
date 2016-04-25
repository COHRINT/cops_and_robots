#!/usr/bin/env python
"""Manages all display elements (figures, axes, etc.)
"""

import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cops_and_robots.fusion.gaussian_mixture import GaussianMixture
from cops_and_robots.display_tools.human_interface import CodebookInterface, ChatInterface

class Display(object):
    """Controls the windows displayed. 

    Each window holds one figure. Each figure holds multiple axes. The Display
    class only creates the window-figure pairs, and assumes that other classes
    handle their figures' axes.
    """
    window_sets = [{'Main': {'size': (14,10),
                             'position': (50,0),
                             },
                   },
                   {'Main': {'size':(13,7),
                             'position': (50,0),
                             },
                    'Chat': {'size':(6,2),
                             'position': (200,700),
                             'is_QtWidget': True,
                             },
                   },
                   {'Main': {'size':(13,7),
                             'position': (50,0),
                             },
                    'Codebook': {'size':(6,2),
                             'position': (100,700),
                             },
                   }
                   ]

    def __init__(self, window_set_id=0, show_vel_interface=False,
                 codebook_interface_cfg={}, chat_interface_cfg={}):

        self._setup_windows(window_set_id)
        self.show_vel_interface = show_vel_interface
        self.codebook_interface_cfg = codebook_interface_cfg
        self.chat_interface_cfg = chat_interface_cfg

    def add_map(self, map_):
        self.map = map_

    def add_vel_states(self, vel_states, vel_space=None):
        self.vel_states = vel_states
        if vel_space == None:
            self.vel_space = np.linspace(-1, 1, 100)
        else:
            self.vel_space = vel_space
        if vel_states is not None:
            self._setup_velocity_axes()

    def add_human_interface(self, human_sensor, questioner=None):
        if "Chat" in self.windows:
            self._setup_chat(human_sensor)
        elif "Codebook" in self.windows:
            self._setup_codebook(human_sensor)
        else:
            logging.debug("No human interface windows available.")
            return

        # Print questions and answers
        if questioner is not None:
            self.questioner = questioner

    def _setup_chat(self, human_sensor):
        # fig = self.windows['Chat']['fig']
        self.chat = ChatInterface(human_sensor, 
                                  **self.chat_interface_cfg)

    def _setup_codebook(self, human_sensor):
        fig = self.windows['Codebook']['fig']
        self.codebook = CodebookInterface(fig, human_sensor,
                                          **self.codebook_interface_cfg)

    def _setup_windows(self, window_set_id):
        window_set = Display.window_sets[window_set_id]
        self.windows = {}

        for name, window in window_set.iteritems():
            try:
                if window['is_QtWidget']:
                    self.windows[name] = window
                    continue
            except:
                logging.debug('Not a QT widget.')

            window['fig'] = plt.figure(figsize=window['size'])
            window['fig'].canvas.set_window_title(name)
            self.windows[name] = window

            # Position windows assuming Qt4 backend
            mngr = plt.get_current_fig_manager()
            geom = mngr.window.geometry()
            x,y,w,h = geom.getRect()
            new_x, new_y = window['position']
            mngr.window.setGeometry(new_x, new_y, w, h)

    def _setup_velocity_axes(self):
        self.vel_axes = {}
        for name, ax in self.map.axes.iteritems():
            #<>TODO: fix for multiple figures!

            # Create colormap Gaussian axes
            w = 0.02
            b = ax.get_position()
            x_box = [None] * 4; y_box = [None] * 4  # left, bottom, width, height
            x_box[0] = b.x0
            x_box[1] = b.y1 * 0.9
            x_box[2] = b.width
            x_box[3] = w
            y_box[0] = b.x1 * 1.02
            y_box[1] = b.x0 * 1.743
            y_box[2] = w
            y_box[3] = b.height * 0.705
            vx_ax = self.windows['Main'].add_axes(x_box)
            vy_ax = self.windows['Main'].add_axes(y_box)

            # Place ticks and labels
            vx_ax.xaxis.tick_top()
            vx_ax.set_xlabel('x velocity (m/timestep)')
            vx_ax.xaxis.set_label_position('top') 
            vx_ax.get_yaxis().set_visible(False)

            vy_ax.yaxis.tick_right()
            vy_ax.set_ylabel('y velocity (m/timestep)')
            vy_ax.yaxis.set_label_position('right') 
            vy_ax.get_xaxis().set_visible(False)


            # # Create 1d Gaussian axes
            # w = 0.06
            # b = ax.get_position()
            # x_box = [None] * 4; y_box = [None] * 4  # left, bottom, width, height
            # x_box[0] = b.x0
            # x_box[1] = b.y1 * 0.868
            # x_box[2] = b.width
            # x_box[3] = w
            # y_box[0] = b.x1
            # y_box[1] = b.x0 * 1.743
            # y_box[2] = w
            # y_box[3] = b.height * 0.705
            # vx_ax = self.windows['Main'].add_axes(x_box)
            # vy_ax = self.windows['Main'].add_axes(y_box)

            # # Place important axes
            # vx_ax.xaxis.tick_top()
            # vx_ax.set_xlabel('x velocity (m/s)')
            # vx_ax.xaxis.set_label_position('top') 

            # vy_ax.yaxis.tick_right()
            # vy_ax.set_ylabel('y velocity (m/s)')
            # vy_ax.yaxis.set_label_position('right') 

            # # Hide other axes
            # vx_ax.get_yaxis().set_visible(False)
            # vx_ax.set_axis_bgcolor([0,0,0,0])
            # vx_ax.spines['bottom'].set_visible(False)
            # vx_ax.spines['left'].set_visible(False)
            # vx_ax.spines['right'].set_visible(False)

            # vy_ax.get_xaxis().set_visible(False)
            # vy_ax.set_axis_bgcolor([0,0,0,0])
            # vy_ax.spines['bottom'].set_visible(False)
            # vy_ax.spines['left'].set_visible(False)
            # vy_ax.spines['top'].set_visible(False)


            self.vel_axes[name] = {'vx': vx_ax,
                                   'vy': vy_ax,
                                   }

    def remove_velocity(self):
        for _, axes in self.vel_axes.iteritems():
            vx_ax = axes['vx']
            vy_ax = axes['vy']
            try:
                vx_contourf = axes['vx_contourf']
                vy_contourf = axes['vy_contourf']
            except:
                logging.error('Exception!')

            # # Remove lines
            # try:
            #     for line in vx_ax.lines:
            #         line.remove()
            #     for line in vy_ax.lines:
            #         line.remove()
            # except:
            #     logging.error('Exception!')

            # Remove fill
            try:
                for coll in vx_contourf.collections:
                    coll.remove()
                del vx_contourf
                for coll in vy_contourf.collections:
                    coll.remove()
                del vy_contourf
            except:
                logging.error('Exception!')

    def update_velocity(self, i):
        # <>TEST STUB
        if np.random.random() < 0.9 and i > 1:
            return

        if i > 1:
            self.remove_velocity()

        for name, axes in self.vel_axes.iteritems():
            vx_ax = axes['vx']
            vy_ax = axes['vy']

            v = self.vel_space
            vx = self.vel_states[name].probability.marginal_pdf(x=v, axis=0)
            vy = self.vel_states[name].probability.marginal_pdf(x=v, axis=1)

            # # TEST STUB
            # vx = GaussianMixture(np.random.random(2), (0.5 - np.random.random(2)), np.random.random(2)).pdf(v)
            # vy = GaussianMixture(np.random.random(2), (0.5 - np.random.random(2)), np.random.random(2)).pdf(v)


            levels = np.linspace(0, np.max((vx, vy)), 80)
            vx = np.tile(vx, (2,1))
            vy = np.tile(vy, (2,1))
            alpha = 0.8
            vx_contourf = vx_ax.contourf(v, [0,1], vx, cmap='viridis', 
                                              levels=levels, alpha=alpha,
                                              antialiased=True, lw=0)
            vy_contourf = vy_ax.contourf([0,1], v, vy.T, cmap='viridis',
                                              levels=levels, alpha=alpha,
                                              antialiased=True, lw=0)
            self.vel_axes[name]['vx_contourf'] = vx_contourf
            self.vel_axes[name]['vy_contourf'] = vy_contourf
            
            # # This is the fix for the white lines between contour levels
            # for c in vx_contourf.collections:
            #     c.set_edgecolor("face")
            #     c.set_alpha(alpha)
            # for c in vy_contourf.collections:
            #     c.set_edgecolor("face")
            #     c.set_alpha(alpha)

            # vx_ax.plot(v, vx, lw=2, color='g')
            # vx_ax.fill_between(v, 0, vx, color='g', alpha=0.2)
            # vy_ax.plot(vy, v, lw=2, color='g')
            # vy_ax.fill_betweenx(v, 0, vy, color='g', alpha=0.2)

    def update_question_answer(self):
        if hasattr(self.questioner, 'recent_answer'):
            str_ = self.questioner.recent_question + ' ' \
                + self.questioner.recent_answer

            if hasattr(self, 'q_text'):
                self.q_text.remove()

            bbox = {'facecolor': 'white',
                    'alpha': 0.8,
                    'boxstyle':'round',
                    }
            ax = self.windows['Main']['fig'].get_axes()[0]
            self.q_text = ax.annotate(str_, xy=(-5, -4.5), 
                                      xycoords='data', annotation_clip=False,
                                      fontsize=16, bbox=bbox)

    def update(self, i):
        try:
            self.map.update(i)
        except AttributeError, e:
            logging.exception("Map not yet defined!")
            raise e

        if hasattr(self, 'vel_states'):
            self.update_velocity(i)

        if hasattr(self, 'questioner'):
            self.update_question_answer()


if __name__ == '__main__':
    from cops_and_robots.map_tools.map import Map
    display = Display(window_set_id=0)
    map_ = Map()
    map_.setup_plot(fig=display.windows['Main']['fig'])
    display.add_map(map_)
    plt.show()