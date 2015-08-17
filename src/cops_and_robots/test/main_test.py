import os
import cProfile
from cops_and_robots.main import main

test_name = 'baseline'

def test_gmm_combined():
    profile_output = os.path.dirname(__file__) \
        + '/../test/profiles/{}/main.prof'.format(test_name)
    cProfile.runctx('main()', globals(), locals(), profile_output)
