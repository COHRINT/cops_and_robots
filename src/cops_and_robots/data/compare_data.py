from __future__ import division
import numpy as np
import logging
import matplotlib.pyplot as plt

from cops_and_robots.fusion.gaussian_mixture import GaussianMixture

def load_posteriors(method_name, trial_name='', sensors_used='human_camera'):

    posteriors = {}
    for measurement_i in range(100):
        try:
            measurement_i = str(measurement_i)
            filename = "ACC 2016/" + sensors_used + "/" + method_name + trial_name + '/'\
                + method_name + '_Pris_posterior_' + measurement_i + '.npy'
            posterior = np.load(filename)

            if type(np.atleast_1d(posterior)[0]) is GaussianMixture:
                posterior = np.atleast_1d(posterior)[0]

            # print filename
            posteriors[measurement_i] = posterior
            # if trial_name != '':
            #     print method_name + trial_name + '_' +  measurement_i
            #     print posterior
        except:
            pass

    return posteriors

def load_MAPs(method_name, trial_name='', sensors_used='human_camera'):
    MAPs = []
    for frame in range(10000):
        try:
            frame = str(frame)
            filename = "ACC 2016/MAP_" + sensors_used + "/" + method_name + trial_name + '/'\
                + method_name + '_Pris_MAP_' + frame + '.npy'
            MAP = np.load(filename)

            # print MAP
            MAPs.append(np.atleast_1d(MAP))
            # if trial_name != '':
            #     print method_name + trial_name + '_' +  measurement_i
            #     print posterior
        except:
            pass

    MAPs = np.array(MAPs)
    return MAPs

def compare_posteriors(test_posterior, true_posterior):


    bounds = [-9.5, -3.33, 4, 3.68]
    grid_size = 0.1
    X, Y = np.mgrid[bounds[0]:bounds[2]:grid_size,
                    bounds[1]:bounds[3]:grid_size]
    pos = np.empty(X.shape + (2,))
    pos = np.dstack((X, Y))
    pos = np.reshape(pos, (X.size, 2))

    if type(test_posterior) is GaussianMixture:
        if not hasattr(test_posterior,'pos'):
            test_posterior._discretize(bounds)
        q_i = test_posterior.pdf(pos)
    else:
        q_i = test_posterior

    if type(true_posterior) is GaussianMixture:
        if not hasattr(true_posterior,'pos'):
            true_posterior._discretize(bounds)
        p_i = true_posterior.pdf(pos)
    else:
        p_i = true_posterior

    p_i /= p_i.sum()
    q_i /= q_i.sum()

    grid_spacing = 0.1
    ndims = 2
    kld = np.nansum(p_i * np.log(p_i / q_i)) * grid_spacing ** ndims

    return kld

def plot_MAPs(MAPs, sensors_used):
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(111)

    colors = {'grid': 'black',
              'recursive': 'red',
              'full_batch': 'green',
              'windowed_batch_1': 'orange',
              'windowed_batch_2': 'cornflowerblue',
              'windowed_batch_2_trial2': 'royalblue',
              'windowed_batch_2_trial3': 'blue',
              'windowed_batch_5': 'purple',
              'no_measurement': 'green'
                  }
    markers = {'grid': '.',
               'recursive': 'x',
               'full_batch': 'o',
               'windowed_batch_1': '^',
               'windowed_batch_2': 's',
               'windowed_batch_2_trial2': 's',
               'windowed_batch_2_trial3': 's',
               'windowed_batch_5': 'p',
               'no_measurement': '*',
               }

    for method, MAP in MAPs.iteritems():
        ms = 0
        if method == 'grid':
            ms *= 2
        x = range(MAP.shape[0])
        y = np.linalg.norm(MAP - np.array([-6,0.5]), ord=2, axis=1)
        # x = MAP[:,0]
        # y = MAP[:,1]
        ax.plot(x, y, ls='-', linewidth=4, label=method, markersize=ms,
                color=colors[method], marker=markers[method],
                alpha=0.9,
                )
        # ax.set_aspect('equal')

    ax.set_xlabel('x (meters)')
    ax.set_ylabel('y (meters)')
    if sensors_used == 'human_camera':
        ax.set_title('Greedy Path for each Fusion Method (human sensor + camera)')
    else:
        ax.set_title('Greedy Path for each Fusion Method (human sensor only)')
    plt.legend()
    plt.show()

def plot_comparison(method_klds, sensors_used='human_camera'):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    markers = {'recursive': 'x',
               'full_batch': 'o',
               'windowed_batch_1': '^',
               'windowed_batch_2': 's',
               'windowed_batch_2_trial2': 's',
               'windowed_batch_2_trial3': 's',
               'windowed_batch_5': 'p',
               'no_measurement': '*',
               }
    colors = {'recursive': 'red',
              'full_batch': 'green',
              'windowed_batch_1': 'orange',
              'windowed_batch_2': 'cornflowerblue',
              'windowed_batch_2_trial2': 'blue',
              'windowed_batch_2_trial3': 'darkblue',
              'windowed_batch_5': 'purple',
              'no_measurement': 'green'
              }

    handles = []
    labels = []
    for method, klds in method_klds.iteritems():
        y = klds.values()
        x = [int(measurement_i) for measurement_i in klds.keys()]
        label = method.title().replace('_',' ')
        label = label.replace('Trial2','(Trial 2)')
        label = label.replace('Trial3','(Trial 3)')
        label = label.replace('Batch 1',r'Batch $\omega=1$')
        label = label.replace('Batch 2',r'Batch $\omega=2$')
        label = label.replace('Batch 5',r'Batch $\omega=5$')

        h = ax.scatter(x,y, marker=markers[method], s=100, lw=2.5, facecolor='none', 
                   edgecolor=colors[method], label=label)
        handles.append(h)
        labels.append(label)

    # # Sort legend
    # labels, handles = zip(*sorted(zip(labels, handles)))

    plt.legend(loc=2, handles=handles, labels=labels)
    ax.set_ylim([0,0.15])
    if sensors_used == 'human_camera':
        ax.set_title('Accuracy of fusion strategies (human sensor + camera)', fontsize=16)
    else:
        ax.set_title('Accuracy of fusion strategies (human sensor only)', fontsize=16)
    ax.set_xticks(range(0,20,2))
    ax.set_xlabel('Measurement number', fontsize=16)
    ax.set_ylabel('KLD (nats) with respect to grid fusion', fontsize=16)

    plt.tight_layout()
    plt.show()

def compare_all(sensors_used='human_only'):
    methods = ['grid',
               'recursive',
               # 'full_batch',
               'windowed_batch_1',
               'windowed_batch_2',
               'windowed_batch_2',  # trial2
               'windowed_batch_2',  # trial3
               'windowed_batch_5',
               # 'no_measurement',
               ]
    trials = ['',
              '',
              # 'full_batch',
              '',
              '',
              '_trial2',
              '_trial3',
              '',
              # '',
              ]
    posteriors = {}
    for i, method in enumerate(methods):
        post = load_posteriors(method, trials[i], sensors_used=sensors_used)
        if len(post) > 0:
            posteriors[method + trials[i]] = post
    klds = {'recursive':{},
            # 'full_batch':{},
            'windowed_batch_1':{},
            'windowed_batch_2':{},
            'windowed_batch_2_trial2':{},
            'windowed_batch_2_trial3':{},
            'windowed_batch_5':{},
            # 'no_measurement':{},
            }
    
    for i, method in enumerate(methods[1:]):
        
        test_posterior = posteriors[method + trials[i+1]]
        true_posterior = posteriors['grid']

        for measurement_i, posterior in test_posterior.iteritems():
            try:
                true_posterior_at_measurement_i = posteriors['grid'][measurement_i]
                kld = compare_posteriors(posterior, true_posterior_at_measurement_i)
                klds[method + trials[i+1]][measurement_i] = kld
            except KeyError:
                logging.error('No shared posterior at measurement_i {}.'.format(measurement_i))

    # print klds
    plot_comparison(klds, sensors_used)

    # Compare MAPs
    MAPs = {}
    for i, method in enumerate(methods):
        MAP = load_MAPs(method, trials[i], sensors_used=sensors_used)
        if MAP.size > 0:
            MAPs[method + trials[i]] = MAP

    # plot_MAPs(MAPs, sensors_used)


if __name__ == '__main__':
    compare_all('human_camera')