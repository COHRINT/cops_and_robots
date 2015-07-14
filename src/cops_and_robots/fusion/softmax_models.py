#!/usr/bin/env python
"""Collection of softmax models

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

# Avoiding circular imports by import
import cops_and_robots.fusion.softmax_class as softmax_class
import cops_and_robots.fusion as fusion
import cops_and_robots.fusion.softmax as softmax
import cops_and_robots.fusion.binary_softmax as binary_softmax
import cops_and_robots.fusion.multimodal_softmax as multimodal_softmax

import numpy as np
import logging
from shapely.geometry import box, Polygon

