from __future__ import division
import logging
import numpy as np
import matplotlib.pyplot as plt
import itertools

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
                ax = fig.add_subplot(111, projection='3d')
            else:
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

        ax.axis('scaled')
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Class Probabilities')

    def find_class_neighbours(self):
        """Method of a Softmax class to find its neighbour classes.

        Applies to subclasses as well as classes.
        """
        from _synthesis import generate_inequalities, find_redundant_constraints
        G, h = generate_inequalities(self.softmax_collection, self.label)
        results, _ = find_redundant_constraints(G, h)

        neighbours = []
        i = 0
        for j in range(len(results) + 1):
            if j == self.id:
                continue
            if not results[i]['is redundant']:
                if self.softmax_collection.has_subclasses:
                    label = self.softmax_collection.subclass_labels[j]
                else:
                    label = self.softmax_collection.class_labels[j]
                neighbours.append(label)
            i += 1
        self.neighbours = neighbours

    def find_critical_points(self, bounds=None):
        if not hasattr(self, 'neighbours'):
            self.find_class_neighbours()
        ndims = self.weights.size

        # Get all possible combinations of neighbours
        neighbour_labels = self.neighbours
        if self.softmax_collection.has_subclasses:
            parent_collection = self.softmax_collection.subclasses
        else:
            parent_collection = self.softmax_collection.classes

        if bounds is not None:
            bound_labels = ['Left Bound', 'Top Bound', 'Right Bound', 
                            'Bottom Bound']
            neighbour_labels = neighbour_labels + bound_labels
            bound_normals = np.array([[-1,0],[0,1],[1,0],[0,-1]])
            bound_offsets = np.array([-np.abs(bounds[0]),
                                       -np.abs(bounds[3]),
                                       -np.abs(bounds[2]),
                                       -np.abs(bounds[1]),
                                        ])
        neighbour_comb_labels = list(itertools.combinations(neighbour_labels, ndims))
        
        neighbour_normals = []
        for neighbour_label in neighbour_labels:
            if 'Bound' in neighbour_label:
                b = bound_labels.index(neighbour_label)
                neighbour_normals.append(bound_normals[b])
            else:
                normal = self.weights - parent_collection[neighbour_label].weights
                neighbour_normals.append(normal)
        neighbour_comb_normals = list(itertools.combinations(neighbour_normals, ndims))

        neighbour_offsets = []
        for neighbour_label in neighbour_labels:
            if 'Bound' in neighbour_label:
                b = bound_labels.index(neighbour_label)
                neighbour_offsets.append(bound_offsets[b])
            else:
                offset = parent_collection[neighbour_label].bias - self.bias
                neighbour_offsets.append(offset)
        neighbour_comb_offsets = list(itertools.combinations(neighbour_offsets, ndims))


        self.critical_points = []
        for i, normals in enumerate(neighbour_comb_normals):
            
            # Remove boundary-boundary intersections
            for label in neighbour_comb_labels[i]:
                if 'Bound' not in label:
                    break
            else:
                break
            comb_label = " + ".join([self.label] + list(neighbour_comb_labels[i]))

            normals = np.array(normals)
            offsets = np.array(neighbour_comb_offsets[i])
            A = normals
            B = offsets

            if np.linalg.matrix_rank(A) < ndims:
                continue

            # Find equiprobable point
            try:
                x = np.linalg.solve(A,B)
            except np.linalg.linalg.LinAlgError:
                logging.debug('No solution found for combination {}'
                              .format(comb_label))
                continue

            # Test to see if equiprobable point is a critical point
            x = np.array([x])
            P_D_x = self.probability(state=x)
            tol = 10 ** -6
            # print '\n'
            # print comb_label
            # print P_D_x
            # print '\n'
            for _, class_ in parent_collection.iteritems():
                test_P_D_x = class_.probability(state=x)
                # print test_P_D_x
                if P_D_x + tol < test_P_D_x:
                    # print 'BROKE'
                    break
            else:
                # print comb_label
                self.critical_points.append(x)

        self.critical_points = np.array(self.critical_points).reshape(-1,2)
        return self.critical_points



if __name__ == '__main__':
    np.set_printoptions(precision=10, suppress=True)
    from _models import intrinsic_space_model, range_model

    # <>TODO: extend to MMS as well
    sm = intrinsic_space_model()
    # sm = range_model()
    bounds = [-5,-5,5,5]

    from shapely.geometry import Polygon
    from shapely.affinity import scale
    from descartes.patch import PolygonPatch
    import matplotlib.pyplot as plt
    fig = plt.figure()
    i = 0
    colors = ['black','red','blue','green','orange']

    min_ = 10000
    max_ = -10000
    for _, class_ in sm.classes.iteritems():
        cps = class_.find_critical_points(bounds)

        #<>TODO: fix ROW swap hack (maybe?)
        cps = np.array([[[cps[1].tolist()] +
                             [cps[0].tolist()] +
                             cps[2:].tolist()
                             ]]).reshape(-1,2)

        ax = fig.add_subplot(111)
        poly = Polygon(cps)
        linepoly = scale(Polygon(cps),0.98,0.98)
        patch = PolygonPatch(poly, facecolor=colors[i], zorder=1,
                                 linewidth=0, alpha=0.6)
        ax.add_patch(patch)
        linepatch = PolygonPatch(linepoly, facecolor='none', zorder=2,
                                 linewidth=3, edgecolor=colors[i], alpha=1.0)
        ax.add_patch(linepatch)
        min_ = np.minimum(cps.min(), min_)
        max_ = np.maximum(cps.max(), max_)
        ax.set_xlim([min_, max_])
        ax.set_ylim([min_, max_])
        i += 1
    plt.show()



