#!/usr/bin/env python
"""Provides interface types for the human operator/sensor.

TYPE ME UP

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``particle_filter`` as one method to represent information.
        2. ``gaussian_mixture_model`` as another method.
        3. ``feasible_layer`` to generate feasible particles and/or
           probabilities.
        4. ``shape_layer`` to ground the human sensor's output.
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

from matplotlib.widgets import RadioButtons, Button

# <>TEST STUB:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class HumanInterface(object):
    """docstring for HumanInterface"""

    types = ['radio_buttons', 'textbox']

    def __init__(self, fig, human_sensor=None,
                 robber_names=['Roy', 'Pris'],
                 cop_names=['Deckard'],
                 groundings=['Wall 1', 'Wall 2', 'Wall 3', 'Wall 4'],
                 type_='radio_buttons'):
        # Sort input object lists
        robber_names.sort()
        cop_names.sort()
        groundings.sort()

        # General interface parameters
        self.fig = fig
        self.type = type_
        self.human_input = ''
        self.radio = {}

        # Content
        if human_sensor:
            self.robber_names = human_sensor.robber_names
            self.groundings = human_sensor.groundings
            self.certainties = human_sensor.certainties
            self.relationships = human_sensor.relationships
        else:
            self.robber_names = ['nothing', 'a robber'] + robber_names
            self.groundings = groundings
            self.certainties = ['think', 'know']
            self.relationships = ['behind', 'in front of', 'left of', 'right of']


        # Radio button parameters and default values
        self.radio_boxcolor = None
        self.certainty = self.certainties[0]
        self.target = self.robber_names[0]
        self.relation = self.relationships[0]
        self.grounding = self.groundings[0]

        # General button parameters
        self.button_color = 'lightgreen'
        self.button_color_hover = 'palegreen'

        # Create interface between interface and human sensor
        self.human_sensor = human_sensor

        if self.type == 'radio_buttons':
            self.make_radio_buttons()
        elif self.type == 'textbox':
            self.make_textbox()

    def make_radio_buttons(self):
        self.fig.subplots_adjust(bottom=0.22)

        # Make the input a complete sentence (and pretty! ...ish)
        self.fig.text(0.18, 0.09, 'I')
        self.fig.text(0.45, 0.09, 'is')
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.patch.set_visible(False)
        ax.axis('off')
        ax.add_patch(Rectangle((0.17, 0.02), 0.72, 0.16, fc='white',
                               edgecolor='black', zorder=-100))

        # Certainty radio buttons
        rax = plt.axes([0.2, 0.025, 0.15, 0.15], axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['certain'] = RadioButtons(rax, self.certainties)

        def certain_func(label):
            self.certainty = label
            logging.debug(self.certainty)
        self.radio['certain'].on_clicked(certain_func)

        # Target radio buttons
        rax = plt.axes([0.3, 0.025, 0.15, 0.15], axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['target'] = RadioButtons(rax, self.robber_names)

        def target_func(label):
            self.target = label
            logging.debug(self.target)
        self.radio['target'].on_clicked(target_func)

        # Relationship radio buttons
        rax = plt.axes([0.5, 0.025, 0.15, 0.15], axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['relation'] = RadioButtons(rax, self.relationships)

        def relation_func(label):
            self.relation = label
            logging.debug(self.relation)
        self.radio['relation'].on_clicked(relation_func)

        # Map object radio buttons
        rax = plt.axes([0.65, 0.025, 0.15, 0.15], axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['map_obj'] = RadioButtons(rax, self.groundings)

        def map_obj_func(label):
            self.grounding = label
            logging.debug(self.grounding)
        self.radio['map_obj'].on_clicked(map_obj_func)

        # Submission button
        rax = plt.axes([0.775, 0.07, 0.10, 0.06])
        self.submit_button = Button(rax, 'Submit', color=self.button_color,
                                    hovercolor=self.button_color_hover)

        def submit_selection(label):
            self.human_input = 'I ' + self.certainty + ' ' + self.target + \
                ' is ' + self.relation + ' ' + self.grounding + '.'
            logging.info('Human says: {}'.format(self.human_input))
            
            if self.human_sensor:
                self.human_sensor.input_string = self.human_input
                for str_ in self.robber_names:
                    if str_ in self.human_input:
                        self.human_sensor.target = str_
        self.submit_button.on_clicked(submit_selection)

    def make_textbox(self):
        pass


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.DEBUG,)
    t = np.arange(0.0, 2.0, 0.01)
    s0 = np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    l, = ax.plot(t, s0, lw=2, color='red')

    hi = HumanInterface(fig)

    plt.show()
