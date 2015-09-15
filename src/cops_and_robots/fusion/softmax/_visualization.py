from __future__ import division
import logging
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # Seemingly unused, but for 3D plots
from matplotlib.colors import ColorConverter
from shapely.geometry import box, Polygon

from cops_and_robots.helpers.visualizations import plot_multisurface

# The following are class methods of Softmax

def plot(self, class_=None, show_plot=True, plot_3D=True, plot_probs=True,
         plot_dominant_classes=True, plot_poly=False, plot_normals=False,
         plot_subclasses=False, plot_legend=True, fig=None, ax=None,
         title='Softmax Classification',
         **kwargs):
    """Display the class and/or PDF plots of the Softmax distribution.

    The class plot shows only the critical classes (those that have the
    greatest probability at any given state).

    Parameters
    ----------
    plot_dominant_classes : bool, optional
        Plot the critical classes. Defaults to `True`.
    plot_probs : bool, optional
        Plot the probability densities. Defaults to `True`.
    plot_poly : bool, optional
        Plot the polygon from which the boundaries are formed. Defaults to
        `False`.
    **kwargs
        Keyword arguments for ``plot_dominant_classes``.
    """

    # Define probabilities lazily
    if not hasattr(self, 'probs') and not plot_subclasses:
            self.probability()
    if not hasattr(self, 'subclass_probs') and plot_subclasses:
            self.probability(find_subclass_probs=True)

    # Plotting attributes
    self.plot_3D = plot_3D
    self.plot_subclasses = plot_subclasses

    if plot_dominant_classes and plot_probs and class_ is None:
        if fig is None:
            self.fig = plt.figure(figsize=(14, 8))
        else:
            self.fig = fig
        bbox_size = (-1.3, -0.175, 2.2, -0.075)

        if ax is None:
            ax1 = self.fig.add_subplot(1, 2, 1)
            if plot_3D and self.state.shape[1] > 1:
                ax2 = self.fig.add_subplot(1, 2, 2, projection='3d')
            else:
                ax2 = self.fig.add_subplot(1, 2, 2)
        else:
            ax1 = ax[0]
            ax2 = ax[1]
        self._plot_dominant_classes(ax1)
        self._plot_probs(ax2)
        axes = [ax1, ax2]

    elif plot_dominant_classes and class_ is None:
        if fig is None:
            self.fig = plt.figure(figsize=(8, 8))
        else:
            self.fig = fig
        if ax is None:
            ax1 = self.fig.add_subplot(111)
        else:
            ax1 = ax
        bbox_size = (0, -0.175, 1, -0.075)
        self._plot_dominant_classes(ax=ax1, **kwargs)
        axes = [ax1]

    elif plot_probs:
        if fig is None:
            self.fig = plt.figure(figsize=(8, 8))
        else:
            self.fig = fig

        if class_ is not None:
            if ax is None:
                if plot_3D and self.state.shape[1] > 1:
                    ax = self.fig.add_subplot(1, 1, 1, projection='3d')
                else:
                    ax = self.fig.add_subplot(1, 1, 1)
            self.classes[class_].plot(ax=ax, **kwargs)
            axes = [self.fig.gca()]
        else:
            if plot_3D and self.state.shape[1] > 1 and ax is None:
                ax1 = self.fig.add_subplot(111, projection='3d')
            elif ax is None:
                ax1 = self.fig.add_subplot(111)
            else:
                ax1 = ax
            self._plot_probs(ax1, **kwargs)
            axes = [ax1]
        bbox_size = (0, -0.15, 1, -0.05)

    if plot_legend:
        # Create Proxy artists for legend labels
        proxy = [None] * self.num_classes
        for i in range(self.num_classes):
            if self.class_labels[i] not in self.class_labels[:i]:
                proxy_label = self.class_labels[i]
            else:
                proxy_label = "_nolegend_"
            proxy[i] = plt.Rectangle((0, 0), 1, 1, fc=self.class_colors[i],
                                     alpha=0.6, label=proxy_label,)

        plt.legend(handles=proxy, loc='lower center', mode='expand', ncol=4,
                   bbox_to_anchor=bbox_size, borderaxespad=0.)
    plt.suptitle(title, fontsize=16)

    # Plot polygon
    if self.poly is not None and plot_poly and plot_dominant_classes:
        patch = PolygonPatch(self.poly, facecolor='none', zorder=2,
                             linewidth=3, edgecolor='black',)
        ax1.add_patch(patch)

    # Plot normals
    # <>TODO fix crashing issue with vertical normals
    if self.normals is not None and plot_normals and plot_dominant_classes:
        t = np.arange(self.bounds[0], self.bounds[2] + 1)
        for i, normal in enumerate(self.normals):
            if abs(normal[1]) < 0.0001:
                ax1.axvline(self.offsets[i], ls='--', lw=3, c='black')
            else:
                slope = normal[0]
                y = slope * t - self.offsets[i]
                ax1.plot(t, y, ls='--', lw=3, c='black')

    if show_plot:
        plt.show()
    try:
        return axes
    except UnboundLocalError:
        logging.warn('No axes to return.')


def _plot_probs(self, ax=None, class_=None):
    if self.state.shape[1] == 1:
        if ax is None:
            ax = self.fig.gca()
        self._plot_probs_1D(ax, class_)
    elif self.state.ndim == 2:
        if ax is None and self.plot_3D:
            ax = self.fig.gca(projection='3d')
        elif ax is None:
            ax = self.fig.gca()
        self._plot_probs_2D(ax, class_)
    elif self.state.ndim == 3:
        if ax is None:
            ax = self.fig.gca(projection='3d')
        self._plot_probs_3D(ax, class_)
    else:
        raise ValueError('The state vector must be able to be represented '
                         'in 1D, 2D or 3D to be plotted.')

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

def _plot_probs_1D(self, ax, class_):
    if type(class_) is str:
        try:
            class_ = self.classes[class_].id
        except KeyError:
            logging.debug('Couldn\'t find class {}. Looking in subclasses.'
                          .format(class_))
            class_ = self.subclasses[class_].id
            if not hasattr(self, 'subclass_probs'):
                self.probability(find_subclass_probs=True)
            self.plot_subclasses = True
        except e:
            logging.error('Couldn\'t find {} as a class or subclass.'
                           .format(class_))
            raise e
    if self.plot_subclasses:
        Z = self.subclass_probs[:]
    else:
        Z = self.probs[:]


    if class_ is not None:
        ax.plot(self.X[0, :], Z[:,class_], color=self.class_colors[class_])
        ax.fill_between(self.X[0, :], 0, Z[:,class_], color=self.class_colors[class_],
                        alpha=0.4)
    else:
        for i in range(self.num_classes):
            ax.plot(self.X[0, :], Z[:,i], color=self.class_colors[i])
            ax.fill_between(self.X[0, :], 0, Z[:,i], color=self.class_colors[i],
                            alpha=0.4)

    ax.set_xlim(self.bounds[0], self.bounds[2])
    ax.set_ylim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('Probability P(D=i|X)')
    ax.set_title('Class Probabilities')

def _plot_probs_2D(self, ax, class_):
    if class_ is not None:
        if type(class_) is str:
            try:
                class_ = self.classes[class_].id
            except KeyError:
                logging.debug('Couldn\'t find class {}. Looking in subclasses.'
                              .format(class_))
                class_ = self.subclasses[class_].id
                if not hasattr(self, 'subclass_probs'):
                    self.probability(find_subclass_probs=True)
                self.plot_subclasses = True
            except e:
                logging.error('Couldn\'t find {} as a class or subclass.'
                               .format(class_))
                raise e
        if self.plot_subclasses:
            Z = self.subclass_probs[:, class_].reshape(self.X.shape[0], self.X.shape[1])
        else:
            Z = self.probs[:, class_].reshape(self.X.shape[0], self.X.shape[1])
    elif self.plot_subclasses:
        Z = self.subclass_probs.reshape(self.X.shape[0], self.X.shape[1],
                                        self.num_subclasses)
    else:
        Z = self.probs.reshape(self.X.shape[0], self.X.shape[1],
                               self.num_classes)

    if self.plot_3D:

        if class_ is not None:
            ax.plot_surface(self.X, self.Y, Z, cstride=2, rstride=2,
                            linewidth=0, antialiased=False,
                            cmap=plt.get_cmap(self.class_cmaps[class_]))
        else:
            #<>TODO: replace with mlab mayavi plotting
            plot_multisurface(self.X, self.Y, Z, ax, cstride=4, rstride=4)

        ax.set_xlim(self.bounds[0], self.bounds[2])
        ax.set_ylim(self.bounds[1], self.bounds[3])
        ax.set_zlabel('P(D=i|X)')
    else:
        levels = np.linspace(0, np.max(Z), 50)
        # <>TODO: Fix contourf plotting for multiple classes
        for i in range(self.num_classes):
            ax.contourf(self.X, self.Y, Z[:,:,i], levels=levels,
                        cmap=plt.get_cmap(self.class_cmaps[i]), alpha=0.8)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Class Likelihoods')

def _plot_probs_3D(self, ax, class_):
    pass

def _plot_dominant_classes(self, ax=None, plot_poly=False, **kwargs):
    """Plot only the critical classes.

    Critical classes are defined as the classes with highest probability
    for a given state vector `x`.

    """
    # <>TODO: Fix the checking of the state vector to allow, say, 1 x y x^2
    # Plot based on the dimension of the state vector
    if ax is None:
        ax = self.fig.gca()

    if self.state.shape[1] == 1:
        self._plot_dominant_classes_1D(ax, **kwargs)
    elif self.state.ndim == 2:
        self._plot_dominant_classes_2D(ax, **kwargs)
    elif self.state.ndim == 3:
        self._plot_dominant_classes_3D(ax, **kwargs)
    else:
        raise ValueError('The state vector must be able to be represented '
                         'in 1D, 2D or 3D to be plotted.')

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

def _plot_dominant_classes_1D(self, ax, **kwargs):

    if self.plot_subclasses:
        probs = self.subclass_probs
    else:
        probs = self.probs

    res = 1 / self.X.size
    fake_X, fake_Y = np.mgrid[self.bounds[0]:self.bounds[2] + res:res,
                              0:0.1]
    max_pdf_indices = np.argmax(probs, axis=1)
    max_colors = np.take(self.class_colors, max_pdf_indices)
    cc = ColorConverter()
    max_colors_rgb = np.array([cc.to_rgb(_) for _ in max_colors])

    ax.bar(self.X.T, np.ones_like(self.X.T), color=max_colors_rgb,
           linewidth=0, alpha=1)

    ax.set_xlim(self.bounds[0], self.bounds[2])
    ax.set_ylim(0, 0.01)
    ax.set_yticks([])

    ax.set_xlabel('x')
    ax.set_title('Critical Classes')

def _plot_dominant_classes_2D(self, ax, **kwargs):

    if self.plot_subclasses:
        probs = self.subclass_probs
    else:
        probs = self.probs

    # <>TODO: come up with more elegant solution than scatter plot
    # Identify colors of critical classes for each state
    np.set_printoptions(threshold=np.nan)
    max_pdf_indices = np.argmax(probs, axis=1)
    max_colors = np.take(self.class_colors, max_pdf_indices)
    cc = ColorConverter()
    max_colors_rgb = np.array([cc.to_rgb(_) for _ in max_colors])

    ax.scatter(self.X, self.Y, c=max_colors_rgb, marker='s', s=100,
               linewidths=0, alpha=1)

    ax.set_xlim(self.bounds[0], self.bounds[2])
    ax.set_ylim(self.bounds[1], self.bounds[3])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Critical Classes')

def _plot_dominant_classes_3D(self, **kwargs):
    pass