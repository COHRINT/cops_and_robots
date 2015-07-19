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

# <>TEST STUB:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class HumanInterface(object):
    """Generate a human interface on a given figure.

    .. image:: img/classes_Human_interface.png

    Parameters
    ----------
    fig : figure handle
        The figure on which to generate the human interface.
    targets : list of str, optional
        The list of all robbers. Defaults to ['Roy','Pris'].
    cop_names : list of str, optional
        The list of all cops. Defaults to ['Deckard'].
    groundings : dict of object and area lists
        A dict containing all grounding elements for human sensor information
        (i.e., in the phrase 'Roy is next to the bookshelf', `the bookshelf`
        is the grounding). Defaults to
        {'area' :
         ['study', 'billiard room', 'hallway', 'dining room', 'kitchen'],
         'object' : ['Deckard', 'bookshelf', 'chair', 'desk', 'table']
         }
    type_ : {'radio_buttons','textbox'}
        The type of human interface to generate.

    """
    input_types = ['radio_buttons', 'textbox']
    measurement_types = {'velocity': False,
                         'area': True,
                         'object': True
                         }

    def __init__(self, fig, human_sensor=None, input_type='radio_buttons',
                 measurement_types=None):
        if measurement_types is None:
            measurement_types = HumanInterface.measurement_types

        # General interface parameters
        self.fig = fig
        self.input_type = input_type
        self.utterance = ''
        self.radio = {}

        # Use human sensor values if one is provided
        if human_sensor:
            self.certainties = human_sensor.certainties
            self.targets = human_sensor.target_names
            self.groundings = {'object': [], 'area': []}
            for key, value in human_sensor.groundings['object'].iteritems():
                self.groundings['object'].append(key)
            for key, value in human_sensor.groundings['area'].iteritems():
                self.groundings['area'].append(key)
            self.positivities = human_sensor.positivities
            self.relations = human_sensor.relations
            self.movement_types = human_sensor.movement_types
            self.movement_qualities = human_sensor.movement_qualities
            logging.debug('using human_sensor values')
        else:
            self.certainties = ['think', 'know']
            self.targets = ['nothing',
                            'a robber',
                            'Roy',
                            'Pris',
                            'Zhora',
                            ]
            self.positivities = ['is', 'is not']
            self.relations = {'object': ['behind',
                                         'in front of',
                                         'left of',
                                         'right of',
                                         ],
                              'area': ['inside',
                                       'near',
                                       'outside'
                                       ]}
            self.groundings = {'area': ['the study',
                                        'the billiard room',
                                        'the hallway',
                                        'the dining room',
                                        'the kitchen',
                                        'the library'
                                        ],
                               'object': ['Deckard',
                                          'the bookshelf',
                                          'the chair',
                                          'the desk',
                                          'the table',
                                          ]}
            self.movement_types = ['moving', 'stopped']
            self.movement_qualities = ['slowly', 'moderately', 'quickly']
        self.groundings['object'].sort()
        self.groundings['area'].sort()
        self.relations['object'].sort()
        self.relations['area'].sort()
        self.targets[2:] = sorted(self.targets[2:])

        # Radio button parameters and default values
        self.radio_boxcolor = None
        self.set_default_values()

        # General button parameters
        self.button_color = 'lightgreen'
        self.button_color_hover = 'palegreen'

        # Create interface between interface and human sensor
        self.human_sensor = human_sensor

        self.fig.subplots_adjust(bottom=0.32)
        self.set_helpers()
        if self.input_type == 'radio_buttons':
            self.current_dialog = 'position (object)'
            self.make_dialog()
        elif self.input_type == 'textbox':
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
        bax = plt.axes([min_x + 2 * tab_w, min_y + h, tab_w, tab_h])
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

        self.make_base_buttons()

    def make_base_buttons(self):
        # Certainty radio buttons
        min_x = 0.05
        min_y = 0.18

        min_x += 0.01
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.07 - h, w, h],
                       axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['certainty'] = RadioButtons(rax, self.certainties)
        self.radio['certainty'].on_clicked(self.certain_func)

        # Target radio buttons
        min_x += w
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.0435 - h, w, h],
                       axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['target'] = RadioButtons(rax, self.targets)
        self.radio['target'].on_clicked(self.target_func)

        # Positivity radio buttons
        min_x += w + 0.02
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.07 - h, w, h],
                       axisbg=self.radio_boxcolor)
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
        if type_ == 'object':
            a = 0.045
        else:
            a = 0.06
        rax = plt.axes([min_x, min_y + a - h, w, h],
                       axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['relation'] = RadioButtons(rax, self.relations[type_])
        self.radio['relation'].on_clicked(self.relation_func)

        # Map object radio buttons
        min_x += w + 0.04
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.045 - h, w, h],
                       axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['grounding'] = RadioButtons(rax, self.groundings[type_])
        self.radio['grounding'].on_clicked(self.grounding_func)

    def make_movement_dialog(self):
        """Genrate the movement radio button interface.
        """

        min_x = 0.35
        min_y = 0.18

        # Movement type radio buttons
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.07 - h, w, h],
                       axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['movement_type'] = RadioButtons(rax, self.movement_types)
        self.radio['movement_type'].on_clicked(self.movement_type_func)

        # Movement quality radio buttons
        min_x += w + 0.04
        w, h = (0.09, 0.18)
        rax = plt.axes([min_x, min_y + 0.06 - h, w, h],
                       axisbg=self.radio_boxcolor)
        rax.patch.set_visible(False)
        rax.axis('off')
        self.radio['movement_quality'] = RadioButtons(rax,
                                                      self.movement_qualities)
        self.radio['movement_quality'].on_clicked(self.movement_quality_func)

    def remove_dialog(self):
        for radio_name, radio in self.radio.iteritems():
            # if radio_name not in ['certainty', 'target', 'positivity']:
            self.fig.delaxes(radio.ax)
        remove_names = ['relation', 'grounding', 'movement_type',
                        'movement_quality']
        for remove_name in remove_names:
            if remove_name in self.radio:
                del self.radio[remove_name]
                logging.debug('deleted {}'.format(remove_name))

    def set_default_values(self, type_='object'):
        self.certainty = self.certainties[0]
        self.target = self.targets[0]
        self.positivity = self.positivities[0]
        self.relation = self.relations[type_][0]
        self.grounding = self.groundings[type_][0]
        self.movement_type = self.movement_types[0]
        self.movement_quality = self.movement_qualities[0]

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
                self.make_base_buttons()
                self.make_position_dialog('object')
                self.set_default_values('object')
                self.fig.canvas.draw()
                logging.info('Swapped dialog to: {}'
                             .format(self.current_dialog))
            else:
                logging.debug('Attempted to swap from {} to position (object).'
                              .format(self.current_dialog))
        self.set_position_obj_dialog = set_position_obj_dialog

        def set_position_area_dialog(event):
            if self.current_dialog != 'position (area)':
                self.current_dialog = 'position (area)'
                self.remove_dialog()
                self.make_base_buttons()
                self.make_position_dialog('area')
                self.set_default_values('area')
                self.fig.canvas.draw()
                logging.info('Swapped dialog to: {}'
                             .format(self.current_dialog))
            else:
                logging.debug('Attempted to swap from {} to position (area).'
                              .format(self.current_dialog))
        self.set_position_area_dialog = set_position_area_dialog

        def set_movement_dialog(event):
            if self.current_dialog != 'movement':
                self.current_dialog = 'movement'
                self.remove_dialog()
                self.make_base_buttons()
                self.make_movement_dialog()
                self.set_default_values()
                self.fig.canvas.draw()
                logging.info('Swapped dialog to: {}'
                             .format(self.current_dialog))
            else:
                logging.debug('Attempted to swap from {} to movement.'
                              .format(self.current_dialog))
        self.set_movement_dialog = set_movement_dialog

        def submit_selection(event):
            # Create human sensor utterance
            if 'position' in self.current_dialog:
                custom_content = ' '.join([self.relation,
                                           self.grounding,
                                           ])
            elif 'movement' in self.current_dialog:
                # <>TODO: stopped slowly?
                custom_content = ' '.join([self.movement_type,
                                           self.movement_quality,
                                           ])
            else:
                custom_content = ''

            self.utterance = ' '.join(['I',
                                             self.certainty,
                                             self.target,
                                             self.positivity,
                                             custom_content
                                             ]) + '.'
            logging.info('Human says: {}'.format(self.utterance))

            # Send result to human sensor
            if self.human_sensor:
                self.human_sensor.utterance = self.utterance
                self.human_sensor.new_update = True  # <>TODO: interrupt
        self.submit_selection = submit_selection

    def make_textbox(self):
        """Generate the textbox interface.
        """
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
