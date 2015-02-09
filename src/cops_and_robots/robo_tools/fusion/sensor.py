#!/usr/bin/env python
"""Provides an base class for various sensor types.

Since many sensors share parameters and functions, the ``sensor``
module defines these in one place, allowing all sensors to use it as
a superclass.

Note:
    Only cop robots have sensors (for now). Robbers may get hardware
    upgreades in future versions, in which case this would be owned by
    the ``robot`` module instead of the ``cop`` module.

Required Knowledge:
    This module and its classes needs to know about the following
    other modules in the cops_and_robots parent module:
        1. ``map_obj`` to represent sensors as map elements.
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


class Sensor(object):
    """docstring for Sensor

    :param update_rate: freuquency of updates in Hz. None for
        intermittant updates.
    :type update_rate:float or None
    """
    def __init__(self, update_rate, has_physical_dimensions, detection_chance):
        super(Sensor, self).__init__()

        # Define simlated sensor parameters
        self.detection_chance = detection_chance  # P(detect|x), x is in view

        self.update_rate = update_rate  # [hz]
