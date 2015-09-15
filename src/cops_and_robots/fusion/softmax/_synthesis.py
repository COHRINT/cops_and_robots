from __future__ import division
import logging
import numpy as np

def batch_test(visualize=False, create_combinations=False):
    # Create base models
    poly1 = _make_regular_2D_poly(4, max_r=2, theta=np.pi/4)
    poly2 = _make_regular_2D_poly(4, origin=[1,1.5], max_r=2, theta=np.pi/4)
    bounds = [-5, -5, 5, 5]
    sm1 = intrinsic_space_model(poly=poly1, bounds=bounds)
    sm2 = intrinsic_space_model(poly=poly2, bounds=bounds)

    # Create intersection model
    M = sm1.num_classes * sm2.num_classes
    intersection_weights = np.empty((M, sm1.weights.shape[1]))
    intersection_biases = np.empty(M)
    i = 0
    for weight1 in sm1.weights:
        for weight2 in sm2.weights:
            intersection_weights[i] = weight1 + weight2
            i += 1
    i = 0
    for bias1 in sm1.biases:
        for bias2 in sm2.biases:
            intersection_biases[i] = bias1 + bias2
            i += 1
    i = 0
    labels = []
    for label1 in sm1.class_labels:
        for label2 in sm2.class_labels:
            labels.append(label1 + ' + ' + label2)

    sm3 = Softmax(intersection_weights,intersection_biases, labels=labels)

     # Create 'front + interior'
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

def find_redundant_constraints(A_full, b_full, break_index=-1, verbose=False):
    """Determine which constraints effect the feasible region."""
    from cvxopt import matrix, solvers
    result = []
    redundant_constraints = []
    for i, _ in enumerate(A_full):
        if i > break_index and break_index > 0:
            break
        A = np.delete(A_full, i, axis=0)
        b = np.delete(b_full, i)

        # Objective function: max c.x (or min -c.x)
        c = -A_full[i] # use the constraint as the objective basis
        beta = b_full[i] # maximum in the constraint basis

        # Check to make sure c is a dense column matrix

        A = matrix(np.asarray(A,dtype=np.float))
        b = matrix(np.asarray(b,dtype=np.float))
        c = matrix(np.asarray(c,dtype=np.float))
        solvers.options['show_progress'] = False
        sol = solvers.lp(c,A,b)

        optimal_pt = np.asarray(sol['x']).reshape(A_full.shape[1])
        # If dual is infeasible, max is unbounded (i.e. infinity)
        if sol['status'] == 'dual infeasible':
            optimal_val = np.inf
        else:
            optimal_val = -np.asarray(sol['primal objective'])
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

    return result, redundant_constraints

def remove_redundant_constraints(A, b, **kwargs):
    _, redundant_constraints = find_redundant_constraints(A, b, **kwargs)
    A = np.delete(A, redundant_constraints, axis=0)
    b = np.delete(b, redundant_constraints)
    return A, b

def test_find_redundant_constraints(verbose=False, show_timing=True, n_runs=1000):
    """Tested against results of LP method in section 3.2 of [1].

    [1] S. Paulraj and P. Sumathi, "A comparative study of redundant 
    constraints identification methods in linear programming problems,"
    Math. Probl. Eng., vol. 2010.
    """
    A_full = np.array([[2, 1, 1],
                       [3, 1, 1],
                       [0, 1, 1],
                       [1, 2, 1],
                       [-1, 0, 0],
                       [0, -1, 0],
                       [0, 0, -1],
                       ], dtype=np.float)
    b_full = np.array([30,
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

        wrapped = wrapper(find_redundant_constraints, A_full, b_full, break_index, False)
        total_time = timeit.timeit(wrapped, number=n_runs)

    if verbose:
        logging.info('LINEAR PROGRAMMING CONSTRAINT REDUCTION RESULTS \n')
    if show_timing:
        logging.info('Average execution time over {} runs: {}s\n'
              .format(n_runs, total_time / n_runs))
    results, _ = find_redundant_constraints(A_full, b_full, break_index, verbose)

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
    A_full = np.array([[-1, 0],
                       [1, 0],
                       [1, 0],
                       [0, -1],
                       [0, 1],
                       ], dtype=np.float)
    b_full = np.array([2,
                       2,
                       4,
                       1,
                       1,
                       ], dtype=np.float)

    A,b = remove_redundant_constraints(A_full, b_full, verbose=verbose)
    return A, b


def lp_test(measurements, verbose=False, visualize=False):
    poly1 = _make_regular_2D_poly(4, max_r=2, theta=np.pi/4)
    poly2 = _make_regular_2D_poly(4, origin=[1,1.5], max_r=2, theta=np.pi/4)
    bounds = [-5, -5, 5, 5]
    sm1 = intrinsic_space_model(poly=poly1, bounds=bounds)
    sm2 = intrinsic_space_model(poly=poly2, bounds=bounds)

    A1, b1 = generate_inequalities(sm1, measurements[0])
    A2, b2 = generate_inequalities(sm2, measurements[1])
    A_full = np.vstack((A1, A2))
    b_full = np.hstack((b1, b2))

    A, b = remove_redundant_constraints(A_full, b_full, verbose=verbose)
    
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

def generate_inequalities(softmax_model, measurement):
    """Produce inequalities in the form Ax \leq b
    """
    
    # Identify the measurement and index
    for i, label in enumerate(softmax_model.class_labels):
        if label == measurement:
            using_subclasses = False
            break
    else:
        if hasattr(softmax_model,'subclasses'):
            for i, label in enumerate(softmax_model.subclass_labels):
                if label == measurement:
                    using_subclasses = True
                    break 
        logging.error('Measurement not found!')

    # Look at log-odds boundaries
    A = np.empty_like(np.delete(softmax_model.weights, 0, axis=0))
    b = np.empty_like(np.delete(softmax_model.biases, 0))
    k = 0
    for j, weights in enumerate(softmax_model.weights):
        if j == i:
            continue
        # Find the inequalities from the log-odds bounds:
        # log(P(D=i|x )/ P(D=j|x ))) = (w_i - w_j)x + (b_i - b_j) >= 0
        # (w_i - w_j)x  >= -(b_i - b_j)
        # -(w_i - w_j)x  <= (b_i - b_j)
        A[k] = -(softmax_model.weights[i] - softmax_model.weights[j])
        b[k] = (softmax_model.biases[i] - softmax_model.biases[j])
        k += 1
    return A, b

def batch_vs_lp():
    measurements = ['Front', 'Inside']
    sm1 = batch_test(visualize=False)
    sm2 = lp_test(measurements, visualize=False)
    combined_measurements =  [measurements[0] + ' + ' + measurements[1]]* 2
    compare_probs(sm1, sm2, measurements=combined_measurements)

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