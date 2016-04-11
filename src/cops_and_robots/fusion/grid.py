#!/usr/bin/env python
from __future__ import division
"""MODULE_DESCRIPTION"""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from scipy.sparse import csr_matrix
from shapely.geometry import Point

from cops_and_robots.fusion.probability import Probability
from cops_and_robots.fusion.gaussian_mixture import fleming_prior


class Grid(Probability):
    """short description of Grid

    long description of Grid
    
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

    def __init__(self, bounds=[-10, -10, 10, 10], res=0.1, prior='fleming',
                 all_dims=False, is_dynamic=True, max_range=1.0, var=1.0,
                 feasible_region=None, use_STM=True):

        if prior == 'fleming':
            bounds = [-9.5, -3.33, 4, 3.68]

        super(Grid, self).__init__(bounds=bounds, res=res)
        self._discretize(all_dims)
        if feasible_region is not None:
            self.identify_feasible_region(feasible_region)
        self.is_dynamic = is_dynamic
        if is_dynamic:
            self.max_range = max_range
            self.var = var
            self.use_STM = use_STM
            if use_STM:
                self._create_STM()
            else:
                self._create_distance_matrix()
                self.update_transition_probs = True

        if prior == 'fleming':
            self.prob = fleming_prior().pdf(self.pos, dims=[0,1])
            self.prob = np.reshape(self.prob, self.X.shape)
        else:
            self.prob = np.ones_like(self.X)
            self.prob /= self.prob.sum()

            # self.keep_feasible_region()

    def __str__(self):
        try:
            num_states = 1
            for shape in self.pos.shape:
                num_states *= shape
            num_states /= self.pos.shape[-1]
        except:
            num_states = '?'
        return 'Gridded probability ({} states)'.format(int(num_states))

    def measurement_update(self, likelihood, measurement=None, **kwargs):
        """Bayesian update of a prior probability with a sensor likelihood.

        Provide likelihood as either a discretized numpy array or as a softmax
        model with an associated measurement class.

        """
        # Discretize likelihood if given as a softmax object
        if type(likelihood) != np.ndarray:
            likelihood = likelihood.probability(class_=measurement, 
                                                state=self.pos)

        # Perform Bayes' update
        posterior = likelihood * self.prob.flatten()
        posterior /= posterior.sum()
        self.prob = np.reshape(posterior, self.X.shape)

    def dynamics_update(self, n_steps=1, velocity_state=None):
        if self.use_STM is False and self.update_transition_probs:
            logging.info('Updating transition probabilities...')
            self.transition_probs = velocity_state.pdf(self.distance_matrix)
            self.update_transition_probs = False

        if self.is_dynamic:
            posterior = self.prob.flatten()
            for step in range(n_steps):
                # self.state_transition_matrix += 0.1 * np.eye(self.state_transition_matrix.shape[1])
                
                if self.use_STM:
                    posterior = self.state_transition_matrix .dot (posterior)
                else:
                    posterior = self.transition_probs .dot (posterior)

                posterior /= posterior.sum()
            self.prob = posterior.reshape(self.X.shape)
            # print 

    def find_MAP(self, dims=[0,1]):
        """formerly 'max_point_by_grid'
        Assume 2D MAP for now
        """
        pt = np.unravel_index(self.prob.argmax(), self.X.shape)
        MAP_point = np.array([self.X[pt[0],0], self.Y[0,pt[1]]])
        MAP_value = self.prob[pt]

        return MAP_point, MAP_value

    def pdf(self, x=None):

        #<>TODO: specify dimensions for evaluation
        x = np.asarray(x)
        if x.shape[0] < self.ndims:  # evaluating at lower dimensionality
            prob = self.probs
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata

        # <>TODO: If x doesn't align with grid points, interpolate

    def as_grid(self, all_dims=False):
        return self.prob

    def identify_feasible_region(self, feasible_region):
        self.infeasible_states = []
        for state_i, pos in enumerate(self.pos):
            pt = Point(pos)
            if not feasible_region.contains(pt):
                self.infeasible_states.append(state_i)
        

    def keep_feasible_region(self):
        prob = self.prob.copy().flatten()
        prob[self.infeasible_states] = 0
        prob /= prob.sum()
        self.prob = prob.reshape(self.X.shape)


    def _discretize(self, bounds=None, res=None, all_dims=False):
        if res is not None:
            self.res = res

        if bounds is None and self.bounds is None:
            b = [-10, 10]  # bounds in any dimension
            bounds = [[d] * self.ndims for d in b]  # apply bounds to each dim
            self.bounds = [d for dim in bounds for d in dim]  # flatten bounds
        elif self.bounds is None:
            self.bounds = bounds

        # Create grid
        if self.ndims == 1:
            x = np.arange(self.bounds[0], self.bounds[1], res)
            self.x = x
            self.pos = x
        elif self.ndims >= 2:

            logging.debug('Using first two variables as x and y')
            X, Y = np.mgrid[self.bounds[0]:self.bounds[2] + self.res:self.res,
                            self.bounds[1]:self.bounds[3] + self.res:self.res]
            pos = np.empty(X.shape + (2,))
            self.X = X; self.Y = Y
            pos = np.dstack((self.X, self.Y))
            self.pos = np.reshape(pos, (self.X.size, 2))

            if all_dims:
                #<>TODO: use more than the ndims == 4 case
                full_bounds = self.bounds[0:2] + [-0.5, -0.5] \
                    + self.bounds[2:] + [0.5, 0.5]
                v_spacing = 0.1
                grid = np.mgrid[full_bounds[0]:full_bounds[4] + res:res,
                                full_bounds[1]:full_bounds[5] + res:res,
                                full_bounds[2]:full_bounds[6] + v_spacing:v_spacing,
                                full_bounds[3]:full_bounds[7] + v_spacing:v_spacing,
                                ]
                pos = np.empty(grid[0].shape + (4,))
                pos[:, :, :, :, 0] = grid[0]
                pos[:, :, :, :, 1] = grid[1]
                pos[:, :, :, :, 2] = grid[2]
                pos[:, :, :, :, 3] = grid[3]

                self.pos_all = pos
        else:
            logging.error('This should be impossible, a gauss mixture with no variables')
            raise ValueError

    def _create_distance_matrix(self):
        n = self.pos.shape[0]
        directory = os.path.dirname(__file__) + '/STMs'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Try to load a precomputed distance matrix
        if hasattr(self, 'infeasible_states'):
            feas_str = '_feasible'
        else:
            feas_str = ''
        filename = '{}/DM_n{}_r{}_{}.npy'.format(directory, n, self.res,
                                                 feas_str)

        try:
            self.distance_matrix = np.load(filename)
            logging.info('Loaded distance matrix {}.'.format(filename))
            return
        except:
            logging.info('No distance matrix to load for {}, creating... '
                         .format({'resolution': self.res,
                                  'feasible': hasattr(self,'infeasible_states')
                                  }))
            logging.info('\n (takes a while for large state spaces)')

        # Create a Distance matrix
        self.distance_matrix = np.empty((n,n,2))
        for state_i, p in enumerate(self.pos):
            # Identify distance components
            dist = p - self.pos

            # Knock out infeasible cells
            if hasattr(self, 'infeasible_states'):
                # dist[self.infeasible_states] = np.inf
                dist[self.infeasible_states] = 1000

            # Save state transition probabilities from state_i
            self.distance_matrix[:, state_i] = dist

            progress = state_i/n * 100
            if state_i % 100 == 0:
                logging.info('Progress: {:.0f}% complete of {} by {} state transition matrix'
                             .format(progress, n, n))

        # Save
        logging.info('Saved distance matrix as {}.'.format(filename))
        np.save(filename, self.distance_matrix)

    def _create_STM(self):
        n = self.pos.shape[0]
        directory = os.path.dirname(__file__) + '/STMs'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Try to load a precomputed STM
        if hasattr(self, 'infeasible_states'):
            feas_str = '_feasible'
        else:
            feas_str = ''
        filename = '{}/STM_n{}_r{}_v{}{}.npy'.format(directory, n, self.res,
                                                     self.var, feas_str)
        try:
            self.state_transition_matrix = np.load(filename).item()
            logging.info('Loaded STM {}.'.format(filename))
            return
        except:
            logging.info('No state transition matrix to load for {}, creating... '
                         .format({'resolution': self.res,
                                  'var': self.var,
                                  'feasible': hasattr(self,'infeasible_states')
                                  }))
            logging.info('\n (takes a while for large state spaces)')

        # Create a STM
        state_transition_matrix = np.empty((n,n))
        covariance = np.eye(self.pos.shape[-1]) * self.var
        for state_i, p in enumerate(self.pos):

            # Identify nearby cells
            norm = np.linalg.norm(p - self.pos, ord=2, axis=1)
            nearby_cells = np.where(norm < self.max_range)[0]

            # Knock out infeasible cells
            if hasattr(self, 'infeasible_states'):
                nearby_cells = [c for c in nearby_cells
                                if c not in self.infeasible_states]
            
            # Sample a gaussian for the state transition probability
            mean = p
            cell_trans = np.zeros(n)
            for nearby_cell in nearby_cells:
                X = self.pos[nearby_cell]
                cell_trans[nearby_cell] = multivariate_normal.pdf(X, mean, covariance)
            cell_trans /= cell_trans.sum()

            # Save state transition probabilities from state_i
            state_transition_matrix[:, state_i] = cell_trans
            
            progress = state_i/n * 100
            if state_i % 100 == 0:
                logging.info('Progress: {:.0f}% complete of {} by {} state transition matrix'
                             .format(progress, n, n))

        # Sparsify and save
        self.state_transition_matrix = csr_matrix(state_transition_matrix)
        logging.info('Saved STM as {}.'.format(filename))
        np.save(filename, self.state_transition_matrix)


def test_dynamics_update(use_STM=True, res=0.2, speed=0.5, vel_var=0.01):
    import matplotlib.animation as animation
    from cops_and_robots.fusion.gaussian_mixture import velocity_prior

    probability = Grid(use_STM=use_STM, res=res)
    vp = velocity_prior(speed=speed, var=vel_var)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    def dm_update(i):
        probability.dynamics_update(velocity_state=vp)
        title = probability.__str__() + '@ time {}'.format(i)
        probability.update_plot(i, title=title)

    ani = animation.FuncAnimation(fig, dm_update,
                                  frames=xrange(100),
                                  interval=100,
                                  repeat=True,
                                  blit=False
                                  )
    # ani.save('demoanimation.gif', writer='imagemagick', fps=10);
    plt.show()

def test_measurement_update():
    import itertools
    import time
    from cops_and_robots.fusion.camera import Camera
    from descartes.patch import PolygonPatch
    import matplotlib.animation as animation

    probability = Grid()
    camera = Camera()
    camera.viewcone.alpha = 0.8
    camera.viewcone.color = 'none'

    poses = [[0,0,-180],
             [-1,0,-180],
             [-1.5,0,-160],
             [-1.6,0,-120],
             [-1.8,-0.5,-120],
             [-2.4,-0.8,-140],
             [-3.0,-0.8,-180],
            ]
    poses = itertools.cycle(poses)

    measurement = 'No Detection'
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def tm_update(i, poses):
        pose = next(poses)
        camera.update_viewcone(pose)
        poly = camera.detection_model.poly  # for plotting

        likelihood = camera.detection_model
        probability.measurement_update(likelihood, measurement)


        probability.update_plot(i)
        patch = camera.viewcone.get_patch()
        ax.add_patch(patch)

        time.sleep(0.5)
        # patch.remove()


    ani = animation.FuncAnimation(fig, tm_update,
                                  frames=xrange(100),
                                  fargs=[poses],
                                  interval=5,
                                  repeat=True,
                                  blit=False
                                  )
    plt.show()

def uniform_prior(feasible_region=None, use_STM=True):
    bounds = [-9.5, -3.33, 4, 3.68]
    probability = Grid(prior='uniform', bounds=bounds, use_STM=use_STM,
                       feasible_region=feasible_region)
    return probability


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # grid = Grid()
    # grid.plot()
    # plt.show()
    # test_measurement_update()
    test_dynamics_update(use_STM=False, res=0.4, speed=0.2, vel_var=0.01)