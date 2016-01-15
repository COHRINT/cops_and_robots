#!/usr/bin/env python
from __future__ import division
"""

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
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from cops_and_robots.fusion.probability import Probability


class Particles(Probability):
    """short description of Particles

    long description of Particles
    
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
        super(Particles, self).__init__()
        self.param = param
        

def uniform_particle_prior(feasible_region=None):
    raise NotImplementedError