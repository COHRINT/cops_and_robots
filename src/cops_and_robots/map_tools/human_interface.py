#!/usr/bin/env python
"""Provides interface types for the human operator/sensor.

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
from matplotlib.text import Text

# <>TEST STUB:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class HumanInterface(object):
    """Generate a human interface on a given figure.

    Allows humans to provide phrases in the following forms (with default
    examples).

    *Movement*
    I [certainty] [target] is [movement type] [movement quality].
    * Default certainties:
      * think
      * know
    * Default targets:
      * Roy
      * Pris
      * Zhora
    * Default movement types:
      * stopped
      * moving
    * Default movement qualities:
      * slowly
      * along
      * quickly

    .. image:: img/classes_Human_interface.png

    Parameters
    ----------
    fig : figure handle
        The figure on which to generate the human interface.
    robber_names : list of str, optional
        The list of all robbers. Defaults to ['Roy','Pris'].
    cop_names : list of str, optional
        The list of all cops. Defaults to ['Deckard'].
    groundings : dict of object and area lists
        A dict containing all grounding elements for human sensor information
        (i.e., in the phrase 'Roy is next to the bookshelf', `the bookshelf`
        is the grounding). Defaults to 
        {'area' :
         ['study', 'billiard room', 'hallway', 'dining room', 'kitchen'],
         'object' : ['bookshelf', 'chair', 'desk', 'table']
         }
    type_ : {'radio_buttons','textbox'}
        The type of human interface to generate.

    """
    types = ['radio_buttons', 'textbox']

    def __init__(self, fig, human_sensor=None,
                 robber_names=['Roy', 'Pris', 'Zhora'],
                 cop_names=['Deckard'],
                 groundings={'area' :
                             ['study', 'billiard room', 'hallway',
                             'dining room', 'kitchen', 'library'],
                             'object' : 
                             ['bookshelf', 'chair', 'desk', 'table']
                            },
                 type_='radio_buttons'):
        # Sort input object lists
        robber_names.sort()
        cop_names.sort()
        groundings['area'].sort()
        groundings['object'].sort()

        # General interface parameters
        self.fig = fig
        self.type = type_
        self.human_input_str = ''
        self.radio = {}
        self.relations = {}

        # Content
        if human_sensor:
            self.robber_names = human_sensor.robber_names
            self.groundings = human_sensor.groundings
            self.certainties = human_sensor.certainties
            self.relations['object'] = human_sensor.relations['object']
            self.relations['area'] = human_sensor.relations['area']
            self.movement_type = human_sensor.movement_type
            self.movement_quality = human_sensor.movement_quality
        else:
            self.robber_names = ['nothing', 'a robber'] + robber_names
            self.groundings = groundings
            self.certainties = ['think', 'know']
            self.relations['object'] = ['behind', 'in front of', 'left of',
                                        'right of']
            self.relations['area'] = ['inside', 'near', 'outside',]
            self.movement_types = ['stopped', 'moving']
            self.movement_qualities = ['slowly','moderately','quickly']
        self.positivities = ['is', 'is not']

        # Radio button parameters and default values
        self.radio_boxcolor = None
        self.certainty = self.certainties[0]
        self.target = self.robber_names[0]
        self.positivity = self.positivities[0]
        self.relation = self.relations['object'][0]
        self.grounding = self.groundings['object'][0]
        self.movement_type = self.movement_types[0]
        self.movement_quality = self.movement_qualities[0]

        # General button parameters
        self.button_color = 'lightgreen'
        self.button_color_hover = 'palegreen'

        # Create interface between interface and human sensor
        self.human_sensor = human_sensor

        self.fig.subplots_adjust(bottom=0.32)
        self.set_helpers()
        if self.type == 'radio_buttons':
            self.current_dialog = 'position (object)'
            self.make_dialog()
        elif self.type == 'textbox':
            self.make_textbox()

    def make_dialog(self):
        """Make the whole dialog interface.
        """

        # Make white bounding box
        self.dialog_ax = self.fig.add_axes([0, 0, 1, 1])
        self.dialog_ax.patch.set_visible(False)
        self.dialog_ax.axis('off')
        min_x, min_y, w, h = (0.04, 0.035, 0.92, 0.19)
        self.dialog_ax.add_patch(Rectangle((min_x, min_y), w, h, fc='white',
                                 edgecolor='black', zorder=-100))

        # Make top tabs
        tab_w, tab_h = 0.18, 0.04
        bax = plt.axes([min_x, min_y + h, tab_w, tab_h])
        self.position_obj_button = Button(bax, 'Position (Object)',
                                          color=self.button_color,
                                          hovercolor=self.button_color_hover)
        bax = plt.axes([min_x + tab_w, min_y + h, tab_w, tab_h])
        self.position_area_button = Button(bax, 'Position (Area)',
                                          color=self.button_color,
                                          hovercolor=self.button_color_hover)
        bax = plt.axes([min_x + 2*tab_w, min_y + h, tab_w, tab_h])
        self.movement_button = Button(bax, 'Movement',
                                      color=self.button_color,
                                      hovercolor=self.button_color_hover)

        
        self.make_position_dialog('object')

        self.position_obj_button.on_clicked(self.set_position_obj_dialog)
        self.position_area_button.on_clicked(self.set_position_area_dialog)
        self.movement_button.on_clicked(self.set_movement_dialog)

        # Make submit button
        max_x = min_x + w
        max_y = min_y + h
        w, h = 0.10, 0.06
        min_x = max_x - w - 0.04
        min_y = (max_y - min_y) / 2
        bax = plt.axes([min_x, min_y, w, h])
        self.submit_button = Button(bax, 'Submit', color=self.button_color,
                                    hovercolor=self.button_color_hover)
        self.submit_button.ax.patch.set_visible(True)
        self.submit_button.on_clicked(self.submit_selection)

        # Make the input a complete sentence 
        min_x = 0.05
        min_y = 0.18
        self.fig.text(min_x, min_y, 'I')

        # Certainty radio buttons
        min_x += 0.01
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.07 - h, w, h], axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['certainty'] = RadioButtons(rax, self.certainties)
        self.radio['certainty'].on_clicked(self.certain_func)

        # Target radio buttons
        min_x += w
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.0435 - h, w, h], axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['target'] = RadioButtons(rax, self.robber_names)
        self.radio['target'].on_clicked(self.target_func)

        # Positivity radio buttons
        min_x += w + 0.02
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.07 - h, w, h], axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['positivity'] = RadioButtons(rax, self.positivities)
        self.radio['positivity'].on_clicked(self.positivity_func)

    def make_position_dialog(self, type_='object'):
        """Genrate the position radio button interface.
        """

        min_x = 0.35
        min_y = 0.18

        # Relationship radio buttons
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.045 - h, w, h], axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['relation'] = RadioButtons(rax, self.relations[type_])
        self.radio['relation'].on_clicked(self.relation_func)

        min_x += w + 0.03
        self.radio['filler'] = self.fig.text(min_x, min_y, 'the')

        # Map object radio buttons
        min_x += 0.04
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.045 - h, w, h], axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['map_obj'] = RadioButtons(rax, self.groundings[type_])
        self.radio['map_obj'].on_clicked(self.grounding_func)

    def make_movement_dialog(self):
        """Genrate the movement radio button interface.
        """

        min_x = 0.35
        min_y = 0.18

        # Movement type radio buttons
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.07 - h, w, h], axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['movement_type'] = RadioButtons(rax, self.movement_types)
        self.radio['movement_type'].on_clicked(self.movement_type_func)

        # Movement quality radio buttons
        min_x += w + 0.04
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.06 - h, w, h], axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['movement_quality'] = RadioButtons(rax, self.movement_qualities)
        self.radio['movement_quality'].on_clicked(self.movement_quality_func)

    def remove_dialog(self):
        for radio_name, radio in self.radio.iteritems():
            if type(radio) is Text:
                radio.remove()
            elif radio_name not in ['certainty','target','positivity']:
                radio.ax.clear()
                radio.ax.patch.set_visible(False)
                radio.ax.axis('off')
        remove_names = ['filler', 'relation', 'grounding', 'movement_type', 'movement_quality']
        for remove_name in remove_names:
            if remove_name in self.radio:
                del self.radio[remove_name]


    def make_textbox(self):
        """Generate the textbox interface.
        """
        pass

    def set_helpers(self):
        """Set helper functions for buttons and radios.
        """

        def certain_func(label):
            self.certainty = label
            logging.debug(self.certainty)
        self.certain_func = certain_func

        def target_func(label):
            self.target = label
            logging.debug(self.target)
        self.target_func = target_func

        def positivity_func(label):
            self.positivity = label
            logging.debug(self.positivity)
        self.positivity_func = positivity_func

        def relation_func(label):
            self.relation = label
            logging.debug(self.relation)
        self.relation_func = relation_func

        def grounding_func(label):
            self.grounding = label
            logging.debug(self.grounding)
        self.grounding_func = grounding_func

        def movement_type_func(label):
            self.movement_type = label
            logging.debug(self.movement_type)
        self.movement_type_func = movement_type_func

        def movement_quality_func(label):
            self.movement_quality = label
            logging.debug(self.movement_quality)
        self.movement_quality_func = movement_quality_func

        def set_position_obj_dialog(event):
            if self.current_dialog != 'position (object)':
                self.current_dialog = 'position (object)'
                self.remove_dialog()
                self.make_position_dialog('object')
                self.fig.canvas.draw()
                logging.info('Swapped dialog to: {}'
                    .format(self.current_dialog))
        self.set_position_obj_dialog = set_position_obj_dialog

        def set_position_area_dialog(event):
            if self.current_dialog != 'position (area)':
                self.current_dialog = 'position (area)'
                self.remove_dialog()
                self.make_position_dialog('area')
                self.fig.canvas.draw()
                logging.info('Swapped dialog to: {}'
                    .format(self.current_dialog))
        self.set_position_area_dialog = set_position_area_dialog

        def set_movement_dialog(event):
            if self.current_dialog != 'movement':
                self.current_dialog = 'movement'
                self.remove_dialog()
                self.make_movement_dialog()
                self.fig.canvas.draw()
                logging.info('Swapped dialog to: {}'
                    .format(self.current_dialog))
        self.set_movement_dialog = set_movement_dialog

        def submit_selection(event):
            if 'position' in self.current_dialog:
                custom_content = ' '.join([self.relation,
                                           'the',
                                           self.grounding,
                                           ])
            elif 'movement' in self.current_dialog:
                custom_content = ' '.join([self.movement_type,
                                           self.movement_quality,
                                           ])
            else:
                custom_content = ''

            self.human_input_str = ' '.join(['I',
                                             self.certainty,
                                             self.target,
                                             self.positivity,
                                             custom_content
                                            ]) + '.'
            logging.info('Human says: {}'.format(self.human_input_str))

            if self.human_sensor:
                self.human_sensor.input_string = self.human_input_str
                for str_ in self.robber_names:
                    if str_ in self.human_input_str:
                        self.human_sensor.target = str_
        self.submit_selection = submit_selection

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.DEBUG,)
    t = np.arange(0.0, 2.0, 0.01)
    s0 = np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    l, = ax.plot(t, s0, lw=2, color='red')

    hi = HumanInterface(fig)

    plt.show()
