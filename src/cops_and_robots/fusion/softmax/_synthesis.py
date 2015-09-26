from __future__ import division
import logging
import numpy as np
import itertools

from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
from cops_and_robots.fusion.gaussian_mixture import GaussianMixture
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

# PRODUCT MODEL ###############################################################

def product_model(models):
    """Generate a product model from multiple softmax models.
    """
    from softmax import Softmax

    n = len(models)  # number of measurements

    # Figure out how many terms are needed in denominator
    M = 1  # total number of terms
    for sm in models:
        if sm.has_subclasses:
            M *= sm.num_subclasses
        else:
            M *= sm.num_classes

    # Generate lists of all parameters
    #<>TODO: keep this a numpy-only operation, as per:
    # http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    model_weights = []
    model_biases = []
    model_labels = []
    for i, sm in enumerate(models):
        model_weights.append(sm.weights.tolist())
        model_biases.append(sm.biases.tolist())
        if sm.has_subclasses:
            # Use class labels of each subclass
            class_labels = []
            for label in sm.subclass_labels:
                i = label.find('__')
                if i != -1:
                    label = label[:i]
                class_labels.append(label)
            model_labels.append(class_labels)
        else:
            model_labels.append(sm.class_labels)

    # Get all possible combinations of parameters
    weight_combs = list(itertools.product(*model_weights))
    bias_combs = list(itertools.product(*model_biases))
    label_combs = list(itertools.product(*model_labels))

    # Evaluate all combinations of model parameters 
    product_weights = np.empty((M, models[0].weights.shape[1]))
    product_biases = np.empty(M)
    product_labels = []
    for i, _ in enumerate(bias_combs):
        product_weights[i] = np.array(weight_combs[i]).sum(axis=0)
        product_biases[i] = np.array(bias_combs[i]).sum()
        str_ = " + ".join(label_combs[i])
        product_labels.append(str_)

    sm = Softmax(weights=product_weights,
                 biases=product_biases,
                 labels=product_labels,
                 )

    return sm

# GEOMETRIC MODEL #############################################################

def find_redundant_constraints(G_full, h_full, break_index=-1, verbose=False):
    """Determine which constraints effect the feasible region."""
    result = []
    redundant_constraints = []
    feasible = []
    for i, _ in enumerate(G_full):
        if i > break_index and break_index > 0:
            break
        G = np.delete(G_full, i, axis=0)
        h = np.delete(h_full, i)

        # Objective function: max c.x (or min -c.x)
        c = -G_full[i]  # use the constraint as the objective basis
        beta = h_full[i]  # maximum in the constraint basis

        # <>TODO: Check to make sure c is a dense column matrix

        G = matrix(np.asarray(G, dtype=np.float))
        h = matrix(np.asarray(h, dtype=np.float))
        c = matrix(np.asarray(c, dtype=np.float))
        solvers.options['show_progress'] = False
        sol = solvers.lp(c,G,h)

        optimal_pt = sol['x']

        # If dual is infeasible, max is unbounded (i.e. infinity)
        if sol['status'] == 'dual infeasible' or optimal_pt is None:
            optimal_val = np.inf
        else:
            optimal_val = -np.asarray(sol['primal objective'])
            optimal_pt = np.asarray(optimal_pt).reshape(G_full.shape[1])

        if sol['status'] == 'primal infeasible':
            feasible.append(False)
        else:
            feasible.append((True))

        is_redundant = optimal_val <= beta
        if is_redundant:
            redundant_constraints.append(i)
        
        if verbose:
            logging.info('Without constraint {}, we have the following:'.format(i))
            logging.info(np.asarray(sol['x']))
            logging.info('\tOptimal value (z_i) {} at point {}.'
                  .format(optimal_val, optimal_pt))
            logging.info('\tRemoved constraint maximum (b_i) of {}.'.format(beta))
            logging.info('\tRedundant? {}\n\n'.format(is_redundant))
        result.append({'optimal value': optimal_val, 
                       'optimal point': optimal_pt,
                       'is redundant': is_redundant
                       })

    if not all(feasible):
        redundant_constraints = None

    return result, redundant_constraints

def remove_redundant_constraints(G, h, **kwargs):
    """Remove redundant inequalities from a set of inequalities Gx <= h.
    """
    _, redundant_constraints = find_redundant_constraints(G, h, **kwargs)
    if redundant_constraints is None:
        return None, None

    G = np.delete(G, redundant_constraints, axis=0)
    h = np.delete(h, redundant_constraints)

    return G, h

def generate_inequalities(softmax_model, measurement):
    """Produce inequalities in the form Gx <= h
    """

    # Identify the measurement and index
    for i, label in enumerate(softmax_model.class_labels):
        if label == measurement:
            break
    else:
        if softmax_model.has_subclasses:
            for i, label in enumerate(softmax_model.subclass_labels):
                if label == measurement:
                    break 
            # logging.error('Measurement not found!')

    # Look at log-odds boundaries
    G = np.empty_like(np.delete(softmax_model.weights, 0, axis=0))
    h = np.empty_like(np.delete(softmax_model.biases, 0))
    k = 0
    for j, weights in enumerate(softmax_model.weights):
        if j == i:
            continue
        G[k] = -(softmax_model.weights[i] - softmax_model.weights[j])
        h[k] = (softmax_model.biases[i] - softmax_model.biases[j])
        k += 1

    return G, h

def geometric_model(models, measurements, show_comp_models=False, *args, **kwargs):
    """
    Could be MMS or softmax models
    """

    minclass_measurements = []  # Lowest level (class or subclass)
    for i, model in enumerate(models):
        model_minclass_measurements = []

        if model.has_subclasses:
            for subclass_label in model.subclass_labels:
                test_label = measurements[i] + '__'
                if test_label in subclass_label:
                    model_minclass_measurements.append(subclass_label)

            # Has subclasses, but measurement is not a subclass
            if len(model_minclass_measurements) == 0:
                model_minclass_measurements = [measurements[i]]
        else:
            model_minclass_measurements = [measurements[i]]

        minclass_measurements.append(model_minclass_measurements)

    # Find the softmax model from each combination of subclasses
    measurement_combs = list(itertools.product(*minclass_measurements))
    comp_models = []
    for measurement_comb in measurement_combs:
        sm = geometric_softmax_model(models, measurement_comb, *args, **kwargs)
        if sm is not None:
            comp_models.append(sm)

    # Visualize the component models
    if show_comp_models:
        fig = plt.figure()
        s = int(np.ceil(np.sqrt(len(comp_models))))

        hr_translation ={'Near__0': 'Front',
                         'Near__1': 'Left',
                         'Near__2': 'Back',
                         'Near__3': 'Right',
                        }

        for i, comp_model in enumerate(comp_models):
            ax = fig.add_subplot(s,s,i +1)
            comp_model.plot(ax=ax, fig=fig, plot_probs=False, 
                            plot_legend=False, show_plot=False)

            # Print human readable titles
            hr_title = []
            for meas in comp_model.class_labels[0].split(' + '):
                a = meas.find('__')
                if a == -1:
                    hr_title.append(meas)
                else:
                    hr_title.append(hr_translation[meas])

            ax.set_title(" + ".join(hr_title))

    #Change label names
    from softmax import Softmax

    joint_measurement = " + ".join(measurements)
    for i, comp_model in enumerate(comp_models):
        weights = comp_model.weights
        biases = comp_model.biases
        labels = [joint_measurement] + comp_model.class_labels[1:]

        comp_models[i] = Softmax(weights, biases, labels=labels)
        comp_models[i].parent_labels = comp_model.class_labels[0]
        logging.info(comp_model.class_labels)

    if len(comp_models) == 1:
        return comp_models[0]
    else:
        return comp_models

def geometric_softmax_model(models, measurements, verbose=False, state_spec='x y', bounds=None):
    """Generate one softmax model from others using geometric constraints.
    """
    from softmax import Softmax

    # Get the full, redundant set of inequalities from all models
    G_full = []
    h_full = []
    for i, sm in enumerate(models):
        G, h = generate_inequalities(sm, measurements[i])
        G_full.append(G)
        h_full.append(h)
    G_full = np.asarray(G_full).reshape(-1, G.shape[1])
    h_full = np.asarray(h_full).reshape(-1)

    # Remove redundant constraints to get weights and biases
    G, h = remove_redundant_constraints(G_full, h_full, verbose=verbose)
    if G is None:
        return None
    z = np.zeros((G.shape[1]))
    new_weights = np.vstack((z, G))
    new_biases = np.hstack((0, -h))

    # Generate a label for the important class, and generic ones for the rest
    labels = [" + ".join(measurements)]
    for i in range(h.size):
        labels.append('Class ' + str(i + 1))
    sm = Softmax(new_weights, new_biases, labels=labels, state_spec=state_spec,
                 bounds=bounds)
    return sm

# NEIGHBOURHOOD MODEL ###############################################################

def find_neighbours(self, class_=None):
    """Method of a Softmax model to find neighbours for all its classes.
    """
    if self.has_subclasses:
        classes = self.subclasses
    else:
        classes = self.classes

    for label, class_ in classes.iteritems():
        class_.find_class_neighbours()
    # print "{} has neighbours: {}".format(class_.label, class_.neighbours)

def neighbourhood_model(models, measurements, iteration=1):
    """Generate one softmax model from each measurement class' neighbours.

    Called at two separate times. 
    """
    from softmax import Softmax

    neighbourhood_models = []
    for i, model in enumerate(models):
        # Find neighbours for (sub)classes and initialize neighbourhood params
        if iteration == 1:
            model.find_neighbours()  #<>TODO: this should happen offline
        else:
            measurement_class = model.classes[measurements[0]]
            if measurement_class.has_subclasses:
                for _, subclass in measurement_class.subclasses.iteritems():
                    subclass.find_class_neighbours()
            else:
                measurement_class.find_class_neighbours()

        class_label = measurements[i]
        if model.has_subclasses:
            classes = model.subclasses
        else:
            classes = model.classes
        neighbourhood_weights = []
        neighbourhood_biases = []
        neighbourhood_labels = []

        # Find labels associated with (sub)classes
        if model.has_subclasses:
            labels = []
            class_ = model.classes[class_label]
            for subclass_label, subclass in class_.subclasses.iteritems():
                labels.append(subclass_label)
        else:
            labels = [class_label]

        # Find measurement parameters
        for label in labels:
            neighbourhood_weights.append(classes[label].weights)
            neighbourhood_biases.append(classes[label].bias)
            neighbourhood_labels.append(class_label)

        # Find parameters of neighbours to measurement
        unique_neighbour_labels = []
        for label in labels:  # for each (sub)class measurement
            neighbour_labels = classes[label].neighbours
            for neighbour_label in neighbour_labels:

                # Find the neighbour (super)class and its label
                i = neighbour_label.find('__')
                if i != -1:
                    neighbour_class_label = neighbour_label[:i]
                else:
                    neighbour_class_label = neighbour_label
                neighbour_class = model.classes[neighbour_class_label]

                # Add that class to the neighbourhood if it's new
                if neighbour_class_label not in unique_neighbour_labels \
                    and neighbour_class_label != class_label:
                    unique_neighbour_labels.append(neighbour_class_label)

                    if neighbour_class.has_subclasses:
                        n_classes = neighbour_class.subclasses
                    else:
                        n_classes = {neighbour_class_label:neighbour_class}

                    for _, nc in n_classes.iteritems():
                        neighbourhood_weights.append(nc.weights)
                        neighbourhood_biases.append(nc.bias)
                        neighbourhood_labels.append(neighbour_class_label)
        neighbourhood_weights =  np.asarray(neighbourhood_weights)
        neighbourhood_biases =  np.asarray(neighbourhood_biases)

        sm = Softmax(weights=neighbourhood_weights,
                     biases=neighbourhood_biases,
                     labels=neighbourhood_labels
                     )
        neighbourhood_models.append(sm)

    neighbourhood_sm = product_model(neighbourhood_models)
    return neighbourhood_sm

# HELPERS #####################################################################

def prob_difference(models, joint_measurement):
    #<>TODO: use arbitrary bounds

    probs = []
    for model in models:
        prob = model.probability(class_=joint_measurement)
        prob = prob.reshape(101,101)
        del model.probs

        probs.append(prob)

    prob_diff = -probs[0]
    for prob in probs[1:]:
        prob_diff += prob
    prob_diff = prob_diff.reshape(101,101)

    return prob_diff

def compare_probs(sm1, sm2, measurements, visualize=True, verbose=True):

    bounds = [-5, -5, 5, 5]
    res = 0.1

    probs1 = sm1.probability(class_=measurements[0])
    probs1 = probs1.reshape(101,101)
    del sm1.probs
    max_i = np.unravel_index(probs1.argmax(), probs1.shape)
    min_i = np.unravel_index(probs1.argmin(), probs1.shape)
    sm1stats = {'max prob': probs1.max(),
                'max prob coord': np.array(max_i) * res + np.array(bounds[0:2]),
                'min prob': probs1.min(),
                'min prob coord': np.array(min_i) * res + np.array(bounds[0:2]),
                'avg prob': probs1.mean(),
                }

    probs2 = sm2.probability(class_=measurements[1])
    probs2 = probs2.reshape(101,101)
    del sm2.probs
    sm2stats = {'max prob': probs2.max(),
                'max prob coord': np.array(max_i) * res + np.array(bounds[0:2]),
                'min prob': probs2.min(),
                'min prob coord': np.array(min_i) * res + np.array(bounds[0:2]),
                'avg prob': probs2.mean(),
                }

    prob_diff21 = probs2 - probs1
    prob_diff21 = prob_diff21.reshape(101,101)

    diffstats = {'max diff': prob_diff21.max(),
                 'min diff': prob_diff21.min(),
                 'avg diff': prob_diff21.mean()
                 }

    if verbose:
        print 'Exact softmax stats:'
        for key, value in sm1stats.iteritems():
            print('{}: {}'.format(key, value))

        print '\nGeometric softmax stats:'
        for key, value in sm2stats.iteritems():
            print('{}: {}'.format(key, value))
        
        print '\n Difference stats:'
        for key, value in diffstats.iteritems():
            print('{}: {}'.format(key, value))

    # Iterate scaled version of LP-generated softmax
    scales = np.linspace(0.7, 0.9, 101)
    for scale in scales:
        weights = sm2.weights * scale
        biases = sm2.biases * scale
        labels = sm2.class_labels
        sm3 = Softmax(weights,biases, labels=labels)

        # probs3 = sm3.probability(class_=measurements[1])
        probs3 = probs2 * scale
        probs3 = probs3.reshape(101,101)
        # del sm3.probs

        prob_diff31 = np.abs(probs3 - probs1)
        prob_diff31 = prob_diff31.reshape(101,101)

        print('Avg: {}, max: {}, at scale of {}'
              .format(prob_diff31.mean(), prob_diff31.max(), scale))

    if visualize:
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(2,2,1, projection='3d')
        sm1.plot(class_=measurements[0], ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title('Model 1: {}'.format(measurements[0]))
        
        ax = fig.add_subplot(2,2,2, projection='3d')
        sm2.plot(class_=measurements[0], ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title('Model 2: {}'.format(measurements[1]))
        
        ax = fig.add_subplot(2,2,3)
        c = ax.pcolormesh(sm1.X, sm1.Y, prob_diff21)
        ax.set_title("Difference b/w 1 & 2")
        ax.set_xlim(bounds[0],bounds[2])
        ax.set_ylim(bounds[1],bounds[3])

        ax = fig.add_subplot(2,2,4)
        c = ax.pcolormesh(sm1.X, sm1.Y, prob_diff31)
        ax.set_title("Difference b/w 1 & 2 (rescaled)")
        ax.set_xlim(bounds[0],bounds[2])
        ax.set_ylim(bounds[1],bounds[3])

        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
        fig.colorbar(c, cax=cax)
        
        plt.show()

# TESTS #######################################################################

def test_synthesis_techniques(test_set=1, visualize=True, visualize_base=False, use_MMS=False,
                              show_comp_models=False):
    from _models import _make_regular_2D_poly, intrinsic_space_model, range_model

    # Create the softmax models to be combined
    poly1 = _make_regular_2D_poly(4, max_r=2, theta=np.pi/4)
    poly2 = _make_regular_2D_poly(4, origin=[2,1.5], max_r=3, theta=np.pi/4)
    poly3 = _make_regular_2D_poly(4, max_r=4, theta=np.pi/4)
    poly4 = _make_regular_2D_poly(4, max_r=1, origin=[-1.5,0], theta=np.pi/4)
    poly5 = _make_regular_2D_poly(4, max_r=3, origin=[1.5,0],theta=np.pi/4)

    if use_MMS:
        if test_set == 1:
            sm1 = range_model(poly=poly1)
            sm2 = range_model(poly=poly2)
            sm3 = range_model(poly=poly3)
            models = [sm1, sm2, sm3]
            measurements = ['Near', 'Near', 'Inside']
            polygons = [poly1, poly2, poly3]
        elif test_set == 2:
            sm4 = range_model(poly=poly4)
            sm5 = range_model(poly=poly5)
            models = [sm4, sm5]
            measurements = ['Near', 'Inside']
            polygons = [poly4, poly5]
    else:
        if test_set == 1:
            sm1 = intrinsic_space_model(poly=poly1)
            sm2 = intrinsic_space_model(poly=poly2)
            sm3 = intrinsic_space_model(poly=poly3)
            measurements = ['Front', 'Inside', 'Inside']
            models = [sm1, sm2, sm3]
            polygons = [poly1, poly2, poly3]
        else:
            sm4 = intrinsic_space_model(poly=poly4)
            sm5 = intrinsic_space_model(poly=poly5)
            measurements = ['Left', 'Inside',]
            models = [sm4, sm5]
            polygons = [poly4, poly5]
    joint_measurement = " + ".join(measurements)

    if visualize_base:
        fig = plt.figure()
        s = len(models)
        for i, model in enumerate(models):
            ax = fig.add_subplot(1,s,i + 1)
            if model.has_subclasses:
                ps = True
            else:
                ps = False
            model.plot(ax=ax, fig=fig, plot_probs=False, plot_legend=True,
                       show_plot=False, plot_subclasses=ps)
            ax.set_title("{}".format(measurements[i]))

    # Synthesize the softmax models
    logging.info('Synthesizing product model...')
    s = time.time()
    product_sm = product_model(models)
    e = time.time()
    product_time = e - s
    logging.info('Took {} seconds\n'.format((product_time)))

    logging.info('Synthesizing neighbourhood model (iter 1)...')
    s = time.time()
    neighbour_sm = neighbourhood_model(models, measurements)
    e = time.time()
    neighbour_time = e - s
    logging.info('Took {} seconds\n'.format((neighbour_time)))

    logging.info('Synthesizing neighbourhood model (iter 2)...')
    s = time.time()
    neighbour_sm2 = neighbourhood_model([neighbour_sm], [joint_measurement], iteration=2)
    e = time.time()
    neighbour2_time = e - s
    logging.info('Took {} seconds\n'.format((neighbour2_time)))

    logging.info('Synthesizing geometric model...')
    s = time.time()
    geometric_sm_models = geometric_model(models, measurements, show_comp_models=show_comp_models)
    e = time.time()
    geometric_time = e - s
    logging.info('Took {} seconds\n'.format((geometric_time)))

    # Find their differences
    neighbour_diff = prob_difference([product_sm, neighbour_sm], joint_measurement)
    neighbour_diff2 = prob_difference([product_sm, neighbour_sm2], joint_measurement)
    if not type(geometric_sm_models) == list:
        geometric_sm_models = [geometric_sm_models]
    geometric_diff = prob_difference([product_sm] + geometric_sm_models, joint_measurement)

    # Fuse all of them with a normal
    prior = GaussianMixture(1, -np.ones(2), 3*np.eye(2))
    from cops_and_robots.fusion.variational_bayes import VariationalBayes
    vb = VariationalBayes()

    logging.info('Fusing Product model...')
    s = time.time()
    mu, sigma, beta = vb.update(measurement=joint_measurement,
                                likelihood=product_sm,
                                prior=prior,
                                )
    if beta.size == 1:
        logging.info('Got a posterior with mean {} and covariance: \n {}'
                     .format(mu, sigma))
    product_post = GaussianMixture(beta, mu, sigma)
    e = time.time()
    product_fusion_time = e - s
    logging.info('Took {} seconds\n'.format((product_time)))

    logging.info('Fusing Neighbourhood model 1...')
    s = time.time()
    mu, sigma, beta = vb.update(measurement=joint_measurement,
                                likelihood=neighbour_sm,
                                prior=prior,
                                )
    if beta.size == 1:
        logging.info('Got a posterior with mean {} and covariance: \n {}'
                     .format(mu, sigma))
    neighbour_post = GaussianMixture(beta, mu, sigma)
    e = time.time()
    neighbour_fusion_time = e - s
    logging.info('Took {} seconds\n'.format((neighbour_time)))

    logging.info('Fusing Neighbourhood model 2...')
    s = time.time()
    mu, sigma, beta = vb.update(measurement=joint_measurement,
                                likelihood=neighbour_sm2,
                                prior=prior,
                                )
    if beta.size == 1:
        logging.info('Got a posterior with mean {} and covariance: \n {}'
                     .format(mu, sigma))
    neighbour2_post = GaussianMixture(beta, mu, sigma)
    e = time.time()
    neighbour2_fusion_time = e - s
    logging.info('Took {} seconds\n'.format((neighbour2_time)))

    logging.info('Fusing Geometric model...')
    s = time.time()
    mixtures = []
    raw_weights = []
    ems = ['Near + Inside__0', 'Near + Inside__2', 'Near + Inside__3',]
    for i, geometric_sm in enumerate(geometric_sm_models):
        exact_measurements = geometric_sm.parent_labels.split(' + ')

        mu, sigma, beta = vb.update(measurement=joint_measurement,
                                    likelihood=geometric_sm,
                                    prior=prior,
                                    get_raw_beta=True,
                                    # exact_likelihoods=models,
                                    # exact_measurements=exact_measurements,
                                    )
        new_mixture = GaussianMixture(beta, mu, sigma)
        mixtures.append(new_mixture)
        raw_weights.append(beta)

    # Renormalize raw weights
    raw_weights = np.array(raw_weights)
    raw_weights /= raw_weights.sum()

    # if beta.size == 1:
    #     logging.info('Got a posterior with mean {} and covariance: \n {}'
    #                  .format(mu, sigma))
    geometric_post = mixtures[0].combine_gms(mixtures[1:], raw_weights=raw_weights)
    e = time.time()
    geometric_fusion_time = e - s
    logging.info('Took {} seconds\n'.format((geometric_time)))

    # Compute KLDs
    neighbour_kld = neighbour_post.compute_kld(product_post)
    neighbour2_kld = neighbour2_post.compute_kld(product_post)
    geometric_kld = geometric_post.compute_kld(product_post)

    if visualize:
        fig = plt.figure(figsize=(18,10))
        bounds = [-5,-5,5,5]

        # Plot critical regions (and polys on the product model)
        ax = fig.add_subplot(3,4,1)
        product_sm.plot(plot_probs=False, ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title('Product Model ({:.0f} terms, {:.2f}s)'
            .format(product_sm.biases.size, product_time))
        for poly in polygons:
            from shapely.affinity import translate
            poly = translate(poly,-0.25, -0.25)
            patch = PolygonPatch(poly, facecolor='none', zorder=2,
                                  linewidth=1.5, edgecolor='black',)
            ax.add_patch(patch)
        plt.axis('scaled')
        ax.set_xlim(bounds[0],bounds[2])
        ax.set_ylim(bounds[1],bounds[3])

        ax = fig.add_subplot(3,4,2)
        neighbour_sm.plot(plot_probs=False, ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title('Neighbour Model 1 ({:.0f} terms, {:.2f}s)'
            .format(neighbour_sm.biases.size, neighbour_time))
        plt.axis('scaled')
        ax.set_xlim(bounds[0],bounds[2])
        ax.set_ylim(bounds[1],bounds[3])

        ax = fig.add_subplot(3,4,3)
        neighbour_sm2.plot(plot_probs=False, ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title('Neighbour Model 2 ({:.0f} terms, {:.2f}s)'
            .format(neighbour_sm2.biases.size, neighbour2_time))
        plt.axis('scaled')
        ax.set_xlim(bounds[0],bounds[2])
        ax.set_ylim(bounds[1],bounds[3])

        ax = fig.add_subplot(3,4,4)
        geometric_sm.plot(plot_probs=False, ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title('Geometric Model ({:.0f} terms, {:.2f}s)'
            .format(geometric_sm.biases.size, geometric_time))
        plt.axis('scaled')
        ax.set_xlim(bounds[0],bounds[2])
        ax.set_ylim(bounds[1],bounds[3])

        # Plot prior
        ax = fig.add_subplot(3,4,5)
        title = 'Prior Distribution'
        prior.plot(ax=ax, fig=fig, bounds=bounds, title=title, show_colorbar=True)

        # Plot probability differences
        ax = fig.add_subplot(3,4,6)
        plt.axis('scaled')
        c = ax.pcolormesh(product_sm.X, product_sm.Y, neighbour_diff,)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(c, cax)
        ax.set_title("Neighbour1 minus Product")
        ax.set_xlim(bounds[0],bounds[2])
        ax.set_ylim(bounds[1],bounds[3])

        ax = fig.add_subplot(3,4,7)
        plt.axis('scaled')
        c = ax.pcolormesh(product_sm.X, product_sm.Y, neighbour_diff2,)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(c, cax)
        ax.set_title("Neighbour2 minus Product")
        ax.set_xlim(bounds[0],bounds[2])
        ax.set_ylim(bounds[1],bounds[3])

        ax = fig.add_subplot(3,4,8)
        plt.axis('scaled')
        c = ax.pcolormesh(product_sm.X, product_sm.Y, geometric_diff,)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(c, cax)
        ax.set_title("Geometric minus Product")
        ax.set_xlim(bounds[0],bounds[2])
        ax.set_ylim(bounds[1],bounds[3])

        # Plot posteriors
        ax = fig.add_subplot(3,4,9)
        title = 'Product Posterior ({:.2f}s)'\
            .format(product_fusion_time)
        product_post.plot(ax=ax, fig=fig, bounds=bounds, title=title, show_colorbar=True)

        ax = fig.add_subplot(3,4,10)
        title = 'Neighbour 1 Posterior (KLD {:.2f}, {:.2f}s)'\
            .format(neighbour_kld, neighbour_fusion_time)
        neighbour_post.plot(ax=ax, fig=fig, bounds=bounds, title=title, show_colorbar=True)

        ax = fig.add_subplot(3,4,11)
        title = 'Neighbour 2 Posterior (KLD {:.2f}, {:.2f}s)'\
            .format(neighbour2_kld, neighbour2_fusion_time)
        neighbour2_post.plot(ax=ax, fig=fig, bounds=bounds, title=title, show_colorbar=True)

        ax = fig.add_subplot(3,4,12)
        title = 'Geometric Posterior (KLD {:.2f}, {:.2f}s)'\
            .format(geometric_kld, geometric_fusion_time)
        geometric_post.plot(ax=ax, fig=fig, bounds=bounds, title=title, show_colorbar=True)
        
        if use_MMS:
            type_ = 'MMS'
        else:
            type_ = 'Softmax'

        fig.suptitle("{} combination of '{}'"
                     .format(type_, joint_measurement),
                     fontsize=15)
        plt.show()

def product_test(models, visualize=False, create_combinations=False):
    """
    """
    sm3 = product_model(models)

    # Manually create 'front + interior'
    fi_weights = np.array([sm2.weights[0],
                           sm2.weights[1],
                           sm2.weights[2],
                           -sm1.weights[1],
                           sm2.weights[4],
                           sm1.weights[4] - sm1.weights[1],
                           ])
    fi_biases = np.array([sm2.biases[0],
                          sm2.biases[1],
                          sm2.biases[2],
                          -sm1.biases[1],
                          sm2.biases[4],
                          sm1.biases[4] -sm1.biases[1],
                          ])
    sm4 = Softmax(fi_weights, fi_biases)

    # Plotting 
    if visualize:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(2,2,1)
        sm1.plot(plot_poly=True, plot_probs=False, ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title('Model 1')

        ax = fig.add_subplot(2,2,2)
        sm2.plot(plot_poly=True, plot_probs=False, ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title('Model 2')

        ax = fig.add_subplot(2,2,3)
        sm3.plot(plot_poly=True, plot_probs=False, ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title('Exact Intersection')

        ax = fig.add_subplot(2,2,4, projection='3d')
        sm3.plot(class_='Front + Inside', ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title("Likelihood of 'Front + Inside'")
        ax.set_zlabel('P(D=i|X)')
        ax.zaxis._axinfo['label']['space_factor'] = 2.8

        plt.show()

    return sm3

def geometric_model_test(measurements, verbose=False, visualize=False):
    poly1 = _make_regular_2D_poly(4, max_r=2, theta=np.pi/4)
    poly2 = _make_regular_2D_poly(4, origin=[1,1.5], max_r=2, theta=np.pi/4)
    bounds = [-5, -5, 5, 5]
    sm1 = intrinsic_space_model(poly=poly1, bounds=bounds)
    sm2 = intrinsic_space_model(poly=poly2, bounds=bounds)

    A1, b1 = generate_inequalities(sm1, measurements[0])
    A2, b2 = generate_inequalities(sm2, measurements[1])
    G_full = np.vstack((A1, A2))
    h_full = np.hstack((b1, b2))

    A, b = remove_redundant_constraints(G_full, h_full, verbose=verbose)
    
    new_weights = np.vstack(([0,0], A))
    new_biases = np.hstack((0, -b))

    labels = [measurements[0] + ' + ' + measurements[1]]
    for i in range(b.size):
        labels.append('Class ' + str(i + 1))
    sm3 = Softmax(new_weights, new_biases, labels=labels)
    
    if visualize:
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        ax = axes[0]
        sm1.plot(plot_poly=True, plot_probs=False, ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title('Model 1: {}'.format(measurements[0]))
        ax = axes[1]
        sm2.plot(plot_poly=True, plot_probs=False, ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title('Model 2: {}'.format(measurements[1]))
        ax = axes[2]
        sm3.plot(plot_poly=True, plot_probs=False, ax=ax, fig=fig, show_plot=False,
                 plot_legend=False)
        ax.set_title("Synthesis of the two")
        plt.show()

    return sm3

def product_vs_lp():
    measurements = ['Front', 'Inside']
    sm1 = product_test(visualize=False)
    sm2 = geometric_model_test(measurements, visualize=False)
    combined_measurements =  [measurements[0] + ' + ' + measurements[1]]* 2
    compare_probs(sm1, sm2, measurements=combined_measurements)

def test_1D():
    from _models import speed_model
    sm = speed_model()

    geometric_sm = geometric_softmax_model([sm], ['Medium'], state_spec='x', bounds=[0, 0, 0.4, 0.4])
    # print geometric_sm.bounds

    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(1, 3, 1)
    sm.plot(plot_probs=True, plot_dominant_classes=False, fig=fig, ax=ax, plot_legend=False,
            show_plot=False)
    ax.set_title('Speed Model')

    ax = fig.add_subplot(1, 3, 2)
    geometric_sm.plot(plot_probs=True, plot_dominant_classes=False, fig=fig, ax=ax, plot_legend=False,
                      show_plot=False)
    ax.set_title('Speed Model')
    plt.show()

def test_find_redundant_constraints(verbose=False, show_timing=True, n_runs=1000):
    """Tested against results of LP method in section 3.2 of [1].

    [1] S. Paulraj and P. Sumathi, "A comparative study of redundant 
    constraints identification methods in linear programming problems,"
    Math. Probl. Eng., vol. 2010.
    """
    G_full = np.array([[2, 1, 1],
                       [3, 1, 1],
                       [0, 1, 1],
                       [1, 2, 1],
                       [-1, 0, 0],
                       [0, -1, 0],
                       [0, 0, -1],
                       ], dtype=np.float)
    h_full = np.array([30,
                       26,
                       13,
                       45,
                       0,
                       0,
                       0,
                       ], dtype=np.float)
    break_index = 3

    truth_data = [{'optimal value': 21.67,
                  'optimal point': np.array([4.33, 13.00, 0.00]),
                  'is redundant': True,
                  },
                  {'optimal value': 45.,
                  'optimal point': np.array([15., 0.00, 0.00]),
                  'is redundant': False,
                  },
                  {'optimal value': 26,
                  'optimal point': np.array([0.00, 19.00, 7.00]),
                  'is redundant': False,
                  },
                  {'optimal value': 30.33,
                  'optimal point': np.array([4.33, 13.00, 0.00]),
                  'is redundant': True,
                  }]

    if show_timing:
        import timeit
        def wrapper(func, *args, **kwargs):
            def wrapped():
                return func(*args, **kwargs)
            return wrapped

        wrapped = wrapper(find_redundant_constraints, G_full, h_full, break_index, False)
        total_time = timeit.timeit(wrapped, number=n_runs)

    if verbose:
        logging.info('LINEAR PROGRAMMING CONSTRAINT REDUCTION RESULTS \n')
    if show_timing:
        logging.info('Average execution time over {} runs: {}s\n'
              .format(n_runs, total_time / n_runs))
    results, _ = find_redundant_constraints(G_full, h_full, break_index, verbose)

    # Compare with truth
    diffs = []
    for i, result in enumerate(results):
        ovd = result['optimal value'] - truth_data[i]['optimal value']
        opd = result['optimal point'] - truth_data[i]['optimal point']
        isr = result['is redundant'] == truth_data[i]['is redundant']
        diffs.append({'optimal value diff': ovd,
                      'optimal point diff': opd,
                      'redundancies agree': isr})

    logging.info("TRUTH MODEL COMPARISON\n")
    for i, diff in enumerate(diffs):
        logging.info('Constraint {}'.format(i))
        for d, v in diff.iteritems():
            logging.info('{}: {}'.format(d,v))
        logging.info('\n')

def test_box_constraints(verbose=False):
    """Remove a known redundant constraint for a box polytope.

    Constraints:
        -x1 \leq 2
        x1 \leq 2
        x1 \leq 4
        -x2 \leq 1
        x2 \leq 1
    """

    # Define our full set of inequalities of the form Ax \leq b
    G_full = np.array([[-1, 0],
                       [1, 0],
                       [1, 0],
                       [0, -1],
                       [0, 1],
                       ], dtype=np.float)
    h_full = np.array([2,
                       2,
                       4,
                       1,
                       1,
                       ], dtype=np.float)

    A,b = remove_redundant_constraints(G_full, h_full, verbose=verbose)
    return A, b

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    logging.getLogger().setLevel(logging.INFO)


    # test_1D()
    test_synthesis_techniques(test_set=1, use_MMS=True, visualize=True)

    # product_vs_lp()
    # product_test(visualize=True, create_combinations=True)
    # measurements = ['Inside', 'Right']
    # geometric_model_test(measurements, verbose=False, visualize=True)