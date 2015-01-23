#!/usr/bin/env python
"""Provides an abstracted interface to multiple types of data fusion.

By dictating the type of the fusion_engine, a cop using the fusion 
engine doesn't need to call specific functions for particle filters or
Gaussian Mixture Models -- it asks the fusion engine for what it wants 
(generally a goal pose for the cop's planner) and lets the fusion 
engine call whichever type of data fusion it wants to use.

Note:
    Only cop robots have fusion engines (for now). Robbers may get 
    smarter in future versions, in which case this would be owned by 
    the ``robot`` module instead of the ``cop`` module.

Required Knowledge:
    This module and its classes needs to know about the following 
    other modules in the cops_and_robots parent module:
        1. ``particle_filter`` as one method to represent information.
        2. ``gaussian_mixture_model`` as another method.
        3. ``feasible_layer`` to generate feasible particles and/or 
           probabilities.
        4. ``shape_layer`` to ground the human sensor's output.
"""



__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

from cops_and_robots.robo_tools.fusion.particle_filter import ParticleFilter
from cops_and_robots.robo_tools.fusion.gaussian_mixture_model import \
     GaussianMixtureModel


class FusionEngine(object):
    """

    The fusion engine tracks each robber, as well as a *combined* 
    representation of the average target estimate. 

    :param type_: 'discrete' or 'continuous'.
    :param type_: String.
    :param :
    :type :
    """
    def __init__(self,type_,robber_names,feasible_layer,shape_layer,motion_model='simple'):
        super(FusionEngine, self).__init__()
        
        self.type = type_
        self.filters = {}
        self.GMMs = {}
        self.robber_names = robber_names
        
        self.shape_layer = shape_layer

        if self.type is 'discrete':
            for i,name in enumerate(robber_names):
                self.filters[name] = ParticleFilter(name,feasible_layer,motion_model=motion_model)
        elif self.type is 'continuous':
            for i,name in enumerate(robber_names):
                self.GMMs[name] = GaussianMixtureModel(name,feasible_layer)
        else:
            raise ValueError("FusionEngine must be of type 'discrete' or "\
                             "'continuous'.")

    def update(self,current_pose,sensors,robbers):
        """Update fusion_engine agnostic to fusion type.

        :param fusion_engine: a probabalistic target tracker.
        :type fusion_engine: FusionEngine.
        :param feasible_layer: a map of feasible regions.
        :type feasible_layer: FeasibleLayer.
        :returns: goal pose as x,y,theta.
        :rtype: list of floats.
        """

        #Update sensor values (viewcone, selected zone, etc.)
        for sensor in sensors.values():
            sensor.update(current_pose,self.shape_layer)

        #Update probabilities (particle and/or GMM)
        for robber in robbers.values():
            if self.type is 'discrete':
                self.filters[robber.name].update(sensors['camera'],robber.pose)
                if robber.status is 'detected':
                    print('{} detected!'.format(robber.name))
                    self.filters[robber.name].robber_detected(robber.pose)
            else:
                self.GMMs[robber.name].update()
                if robber.status is 'detected':
                    self.GMMs[robber.name].robber_detected(robber.pose)

    def update_combined(self):
        pass