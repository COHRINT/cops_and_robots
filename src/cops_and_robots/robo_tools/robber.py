#!/usr/bin/env python
"""MODULE_DESCRIPTION"""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

from cops_and_robots.robo_tools.robot import Robot

class Robber(Robot):
    """docstring for Robber"""
    def __init__(self, **kwargs):
        super(Robber, self).__init__(**kwargs)
        