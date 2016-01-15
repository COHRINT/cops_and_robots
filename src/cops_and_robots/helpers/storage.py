#!/usr/bin/env python
"""Provides storage and data logging for the Cops and Robots simulation.

"""
import logging
import os

import pandas as pd
import numpy as np
from pandas import HDFStore

class Storage(object):
    """Records information from the simulation.

    Data is recorded as a high-density filesystem format (a Pandas dataframe).
    """

    def __init__(self, record_data=None, use_prefix=True, use_suffix=True,
                 folder='data/', filename='datalog', save_data=True):
        self.file_path = os.path.dirname(os.path.abspath(__file__)) \
            + '/../' + folder
        self.set_filename(filename, use_prefix, use_suffix)

        self.store = HDFStore(self.filename)
        if record_data is None:
            self.records = {'grid probability':'2D', 
                            'robot positions':'all'
                            }
        else:
            self.records = record_data

        self.dfs = {}

    def set_filename(self, filename, use_prefix, use_suffix):
        """Create the filename of the datafile to hold recorded data.
        """
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)

        if use_prefix:
            self.filename_prefix = 'cnr_'
        else:
            self.filename_prefix = ''

        if use_suffix:
            self.filename_suffix = '_trial'
        else:
            self.filename_suffix = ''
        self.filename_extension = '.hd5'

        self.filename = self.file_path + self.filename_prefix + filename \
            + self.filename_suffix + self.filename_extension

        # Check for duplicate files
        i = 0
        while os.path.isfile(self.filename):
            i += 1
            ind = self.filename.find(self.filename_suffix)
            l = len(self.filename_suffix)
            self.filename = self.filename[:ind + l] + '_' + str(i) \
                + self.filename_extension

    def collect_data(self, cops, robbers, distractors):
        # Record data
        d = {}

        for record, record_value in self.records.iteritems():
            if 'robot positions' == record:
                if record_value == 'all':
                    d['Deckard position'] = cops['Deckard'].pose2D.pose
                    d['Roy position'] = robbers['Roy'].pose2D.pose
                else:
                    #<>TODO: record based on name provided
                    pass

            if 'grid probability' == record:
                if record_value == '4D':
                    all_dims = True
                else:
                    all_dims = False
                #<>TODO: assume more than Roy and Deckard
                d['grid probability'] = cops['Deckard'].fusion_engine\
                    .filters['Roy'].probability.as_grid(all_dims)
                d['grid probability'] = d['grid probability'].flatten()

            if 'questions' == record and record_value:
                if hasattr(cops['Deckard'].questioner, 'question_str'):
                    d['questions'] = cops['Deckard'].questioner.question_str
                    cops['Deckard'].questioner.question_str = 'No question'
                else:
                    d['questions'] = 'No question'

            if 'answers' == record and record_value:
                if cops['Deckard'].sensors['human'].utterance:
                    d['answer'] = cops['Deckard'].sensors['human'].utterance
                else:
                    d['answer'] = 'No answer'

            if 'VOI' == record and record_value:
                if hasattr(cops['Deckard'].questioner, 'VOIs'):
                    d['VOI'] = cops['Deckard'].questioner.VOIs
                else:
                    d['VOI'] = np.nan \
                        * np.empty(len(cops['Deckard'].questioner.all_questions))

            if 'ordered_question_list' == record and record_value:
                if hasattr(cops['Deckard'].questioner, 'ordered_question_list') and \
                    not ('ordered_question_list' in self.dfs.keys()):

                    d['ordered_question_list'] = \
                        cops['Deckard'].questioner.ordered_question_list
        return d

    def save_frame(self, frame_i, data, last_frame=False):
        """Store one frame of the input data.

        Expects data as a dict.
        """
        
        for key, value in data.iteritems():
            key = key.lower().replace(' ', '_')

            # Series for strings or string lists, DataFrame otherwise
            if isinstance(value, basestring):
                value = [value]

            try:
                v = value[0]
                is_list_of_strings = isinstance(v, basestring)
            except:
                is_list_of_strings = False

            if is_list_of_strings: 
                try:
                    new_df = pd.Series(value)
                    self.dfs[key] = self.dfs[key].append(new_df)
                except:
                    self.dfs[key] = pd.Series(value)
            else:
                try:
                    new_df = pd.DataFrame(value, columns=[str(frame_i + 1)])
                    self.dfs[key] = self.dfs[key].join(new_df)
                    #<>TODO: this is SLOW because it's copying every frame. Fix it!
                except:
                    self.dfs[key] = pd.DataFrame(value)

            if last_frame:
                self.store.put(key, self.dfs[key])
