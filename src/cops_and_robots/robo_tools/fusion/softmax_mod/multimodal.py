#!/usr/bin/env python
"""Provides creation, maniputlation and plotting of Softmax distributions.

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
import numpy as np
import copy

from cops_and_robots.robo_tools.fusion.softmax import Softmax


# <>TODO: Break out MMS from SoftMax class
class MultiModalSoftmax(object):
    """short description of MultiModalSoftmax

    long description of MultiModalSoftmax
    
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

    def __init__(self,param):
        super(MultiModalSoftmax, self).__init__()
        self.param = param
        