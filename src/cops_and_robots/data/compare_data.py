from __future__ import division
import numpy as np
import logging

def load_data(method_name):

    posteriors = {}
    for step in range(10000):
        try:
            step = str(step)
            filename = "ACC 2016/" + method_name + '/'\
                + method_name + '_Roy_posterior_' + step + '.npy'
            posterior = np.load(filename)

            posteriors[step] = posterior
        except:
            pass

    return posteriors

def compare_posteriors(test_posterior, true_posterior):

    if type(test_posterior) is GaussianMixture:
        if not hasattr(test_posterior,'pos'):
            test_posterior._discretize()
        q_i = test_posterior.pdf(test_posterior.pos)
    else:
        q_i = test_posterior

    if type(true_posterior) is GaussianMixture:
        if not hasattr(true_posterior,'pos'):
            true_posterior._discretize()
        p_i = true_posterior.pdf(true_posterior.pos)
    else:
        p_i = true_posterior

    grid_spacing = 0.1
    ndims = 2
    kld = np.sum(p_i * np.log(p_i / q_i)) * grid_spacing ** ndims
    return kld

def compare_all():
    methods = ['grid','recursive','full_batch','windowed_batch_2']
    posteriors = {}
    for method in methods:
        posteriors[method] = load_data(method)

    klds = {'recursive':{},
            'full_batch':{},
            'windowed_batch_2':{}
            }
    for method in methods[1:]:
        test_posterior = posteriors[method]
        true_posterior = posteriors['grid']

        for step, posterior in test_posterior.iteritems():
            try:
                true_posterior_at_step = posteriors['grid'][step]
                kld = compare_posteriors(posterior, posteriors['grid'])
                klds[method][step] = kld
            except:
                logging.info('No shared posterior at step {}.'.format(step))

    print klds

if __name__ == '__main__':
    compare_all()