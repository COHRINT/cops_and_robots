#!/usr/bin/env python

import os
import cProfile
from cops_and_robots.main import main

test_name = 'baseline'

def test_particle():
    config_file = os.path.dirname(__file__) \
        + '/../test/configs/particle_profiling.yaml'
    profile_output = os.path.dirname(__file__) \
        + '/../test/profiles/{}/particle.prof'.format(test_name)
    cProfile.runctx('main(config_file)', globals(), locals(), profile_output)

def test_gmm():
    config_file = os.path.dirname(__file__) \
        + '/../test/configs/gmm_profiling.yaml'
    profile_output = os.path.dirname(__file__) \
        + '/../test/profiles/{}/gmm.prof'.format(test_name)
    cProfile.runctx('main(config_file)', globals(), locals(), profile_output)

def test_particle_combined():
    config_file = os.path.dirname(__file__) \
        + '/../test/configs/particle_profiling_combined.yaml'
    profile_output = os.path.dirname(__file__) \
        + '/../test/profiles/{}/particle_combined.prof'.format(test_name)
    cProfile.runctx('main(config_file)', globals(), locals(), profile_output)

def test_gmm_combined():
    config_file = os.path.dirname(__file__) \
        + '/../test/configs/gmm_profiling_combined.yaml'
    profile_output = os.path.dirname(__file__) \
        + '/../test/profiles/{}/gmm_combined.prof'.format(test_name)
    cProfile.runctx('main(config_file)', globals(), locals(), profile_output)
