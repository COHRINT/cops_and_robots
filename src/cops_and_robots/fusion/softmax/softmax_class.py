from __future__ import division
import logging
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # Seemingly unused, but for 3D plots



class SoftmaxClass(object):
    """short description of SoftmaxClass

    long description of SoftmaxClass
    
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

    # Load methods from external files
    from _synthesis import find_class_neighbours

    def __init__(self, id_, label, weights, bias, softmax_collection,
                 steepness=1,color='grey', cmap='Greys'):
        self.id = id_
        self.label = label
        self.weights = np.asarray(weights)
        self.bias = np.asarray(bias)
        self.softmax_collection = softmax_collection  #<>TODO: rename to 'parent_model'
        self.steepness = np.asarray(steepness)
        self.color = color
        self.cmap = cmap
        self.ndim = self.weights.shape[0]

        self.subclasses = {}
        self.has_subclasses = False
        self.num_subclasses = 0

    def add_subclass(self, subclass):
        """Add a subclass to this softmax class.
        """
        subclass_label = self.label
        self.has_subclasses = True

        if len(self.subclasses) == 0:
            self.weights = subclass.weights
            self.bias = subclass.bias
        else:
            self.weights = np.vstack((self.weights, subclass.weights))
            self.bias = np.hstack((self.bias, subclass.bias))

        if len(self.subclasses) > 0:

            # Number unnumbered subclass
            for sc_label, sc in self.subclasses.iteritems():
                i = sc_label.find('__')
                if i == -1:
                    old_sc = self.subclasses[sc_label]
                    del self.subclasses[sc_label]
                    sc_label = sc_label + '__0'
                    self.subclasses[sc_label] = sc

            # Number current subclass
            subclass_label = subclass_label + '__' + str(self.num_subclasses)

        self.subclasses[subclass_label] = subclass

        self.num_subclasses += 1

    def probability(self, state=None, find_subclass_probs=False):
        """Find the probability for this class at a given state.

        Parameters
        ----------
        state : array_like
            A a 1-dimensional array containing the state values of all `N`
            states at which to find the probabilities of each class
            specified.

        Returns
        -------
        An array_like object containing the probabilities of the class(es)
        specified.

        """
        p = self.softmax_collection.probability(state=state, class_=self.label,
                                                find_subclass_probs=find_subclass_probs)
        return p

    def plot(self, ax=None, fill_between=True, plot_3D=True, **kwargs):
        if self.ndim < 2:
            self._plot_1D(ax=ax, fill_between=fill_between)
        elif self.ndim == 2:
            self._plot_2D(ax=ax, plot_3D=plot_3D)
        else:
            logging.error('No plot available at higher dimensions than 2!')

    def _plot_1D(self, ax=None, fill_between=True, **kwargs):
        fig = plt.gcf()
        if ax is None:
            ax = fig.add_subplot(111)

        if hasattr(self,'probs'):
            del self.probs
        if not hasattr(self.softmax_collection, 'probs'):
            self.softmax_collection.probability()
        self.probs = self.softmax_collection.probs[:, self.id]
        self.X = self.softmax_collection.X[0,:]

        ax.plot(self.X, self.probs, color=self.color,
                **kwargs)
        if fill_between:
            ax.fill_between(self.X, 0, self.probs,
                            color=self.color, alpha=0.4)

        bounds = self.softmax_collection.bounds
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(0, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability P(D=i|X)')
        ax.set_title('Class Probabilities')

    def _plot_2D(self, ax=None, plot_3D=True, **kwargs):
        fig = plt.gcf()
        if ax is None:
            if plot_3D:
                print HEEEEEEE
                ax = fig.add_subplot(111, projection='3d')
            else:
                print HEYEHYE
                ax = fig.add_subplot(111)

        if hasattr(self,'probs'):
            del self.probs
        if not hasattr(self.softmax_collection, 'probs'):
            self.softmax_collection.probability()

        X = self.softmax_collection.X
        Y = self.softmax_collection.Y
        Z = self.softmax_collection.probs[:, self.id].reshape(X.shape[0], X.shape[1])
        bounds = self.softmax_collection.bounds

        if plot_3D:
            ax.plot_surface(X, Y, Z, cstride=2, rstride=2, linewidth=0,
                            antialiased=False, cmap=plt.get_cmap(self.cmap))

            ax.set_zlabel('Probability P(D=i|X)')
        else:
            levels = np.linspace(0, np.max(Z), 50)
            ax.contourf(X, Y, Z, levels=levels, cmap=plt.get_cmap(self.cmap),
                        alpha=0.8)

        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Class Probabilities')