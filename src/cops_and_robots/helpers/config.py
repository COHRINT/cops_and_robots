#!/usr/bin/env python

import logging
import yaml
import os

def load_config(filename='config.yaml', path=None):
    if path is None:
        path = os.path.dirname(__file__) + '/../configs/' + filename
    try:
        with open(path, 'r') as stream:
            cfg = yaml.load(stream)
    except IOError, e:
        logging.error('Configuration file \'{}\' not found.'.format(path))
        raise e
    return cfg