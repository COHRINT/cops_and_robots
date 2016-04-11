#!/usr/bin/env python
"""Uses CRF++ to generate tagged semantics.
"""
__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import logging
import numpy as np
import textwrap
import subprocess
import os
import re

class Tagger(object):
    """short description of Tagger

    Uses Conditional Random Fields to tag a tokenized document with semantic
    labels.

    """
    def __init__(self, template_file='crf_template.txt',
                 training_file='crf_training.txt',
                 test_file='crf_test.txt',
                 model_file='crf_model.txt',
                 input_file='crf_input.txt',
                 output_file='crf_output.txt',
                 generate_template=False, generate_training_data=False):
        self.data_dir = os.path.dirname(__file__) + '/data/'
        self.template_file = self.data_dir + template_file
        self.training_file = self.data_dir + training_file
        self.test_file = self.data_dir + test_file
        self.model_file = self.data_dir + model_file
        self.input_file = self.data_dir + input_file
        self.output_file = self.data_dir + output_file

        self._generate_crf_model(generate_template, generate_training_data)
        self._train_crf_model()


    def _generate_crf_model(self, generate_template=False, 
                            generate_training_data=False):
        # Generate template, if necessary
        template_exists = os.path.isfile(self.template_file)
        if generate_template or not template_exists:
            self._create_template()

        # Generate training data, if necessary
        training_file_exists = os.path.isfile(self.training_file)
        if generate_training_data or not training_file_exists:
            training_data = generate_fleming_test_data()

            # Save to a file
            with open(self.training_file,'w') as file_:
                for d in training_data:
                    str_ = '\t'.join(d) + '\n'
                    str_ = str_.replace (" ", "_")
                    file_.write(str_)

    # def test_crf_model(self):
    #     test_data = generate_test_data()
    #     with open(self.test_file,'w') as file_:
    #         for d in test_data:
    #             str_ = '\t'.join(d) + '\n'
    #             str_ = str_.replace (" ", "_")
    #             file_.write(str_)

    def _create_template(self, model_i=1):
        model0 = """
                   # Unigram
                   U00:%x[-2,0]
                   U01:%x[-1,0]
                   U02:%x[0,0]
                   U03:%x[1,0]
                   U04:%x[2,0]
                   U05:%x[-1,0]/%x[0,0]
                   U06:%x[0,0]/%x[1,0]
                   
                   # Bigram
                   B"""

        model1 = """
                   # Unigram
                   U00:%x[-2,0]
                   U01:%x[-1,0]
                   U02:%x[0,0]
                   U03:%x[1,0]
                   U04:%x[2,0]
                   U05:%x[-1,0]/%x[0,0]
                   U06:%x[0,0]/%x[1,0]
                                  
                   # Bigram
                   B"""
        models = [model0, model1]
        template = models[model_i]
        
        with open(self.template_file,'w') as file_:
            file_.write(textwrap.dedent(template))

    def _train_crf_model(self):
        cmd = ("crf_learn {} {} {}"
               .format(self.template_file, self.training_file, self.model_file))
        proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        proc.wait()

    def tag_document(self, tokenized_document):
        """Generate semantic tags for each token in a given document
        """
        # Tokenize document if given a string by accident
        if isinstance(tokenized_document, str):
            tokenized_document = re.findall(r"[\w']+|[.,!?;]",
                                            tokenized_document)

        # Generate the input file to CRF++
        self._generate_crfpp_input(tokenized_document)

        # Run CRF++ (and wait for results)
        cmd = ("crf_test -m {} -v1 {}"
               .format(self.model_file, self.input_file, self.output_file))
        proc = subprocess.Popen(cmd.split(), stdout=open(self.output_file, 'w'))
        proc.wait()

        # Grab the results
        results = self._read_crfpp_output()

        # Remove marginal probability of the tag for each tag
        tagged_document = []
        for result in results:
            if len(result) > 0:
                tagged_document.append(result[:-1])
        # tagged_document = [r for r in tagged_document if r[0] not in (',','.','!','?')]

        return tagged_document

    def _generate_crfpp_input(self, tokenized_document):
        """

        """
        input_data = [[s,''] for s in tokenized_document]
        with open(self.input_file, 'w') as file_:
            for d in input_data:
                str_ = '\t'.join(d) + '\n'
                str_ = str_.replace (" ", "_")
                file_.write(str_)

    def _read_crfpp_output(self, check_answers=False):
        with open(self.output_file,'r') as f:
            output = f.read()
            strings = output.split('\n')
            results = []

            for str_ in strings[1:]:
                result = str_.replace('/','\t').split('\t')
                if len(result) < 3:
                    continue

                # Replace underscores
                result = [result[0].replace('_',' ')] + result[1:]

                # Check answers if correct tags available
                if check_answers:
                    if result[-3] == result[-2]:
                        result += [True]
                    else:
                        result += [False]
                results.append(result)
        return results


def generate_test_data():
    data = [['Roy','TARGET' ],
        ['is','POSITIVITY'],
        ['moving','ACTION'],
        ['North','MODIFIER'],
        ['.','NULL'],
        ['That robot','TARGET'],
        ['is','POSITIVITY'],
        ['stopped','ACTION'],
        ['.','NULL'],
        ['Nothing','TARGET'],
        ['is','POSITIVITY'],
        ['next to','SPATIALRELATION'],
        ['the dresser','GROUNDING'],
        ['.','NULL'],
        ['I','NULL'],
        ['don\'t','POSITIVITY'],
        ['see','NULL'],
        ['anything','TARGET'],
        ['near','SPATIALRELATION'],
        ['the desk','GROUNDING'],
        ['.','NULL'],
        ['I think','NULL'],
        ['a robot','TARGET'],
        ['is','POSITIVITY'],
        ['in','SPATIALRELATION'],
        ['the kitchen','GROUNDING'],
        ['.','NULL'],
        ['Pris','TARGET'],
        ['is','POSITIVITY'],
        ['moving','ACTION'],
        ['really quickly','MODIFIER'],
        ['.','NULL'],
        ['The green one','TARGET'],
        ['is','POSITIVITY'],
        ['heading','ACTION'],
        ['over there','GROUNDING'],
        ['.','NULL'],
        ['The red guy','TARGET'],
        ['is','POSITIVITY'],
        ['spinning around','ACTION'],
        ['the table','GROUNDING'],
        ['.','NULL'],
        ['A robot\'s','TARGET'],
        ['moving','ACTION'],
        ['away from','MODIFIER'],
        ['you','GROUNDING'],
        ['.','NULL'],
        ['There\'s','NULL'],
        ['another robot','TARGET'],
        ['heading','ACTION'],
        ['towards','MODIFIER'],
        ['you','GROUNDING'],
        ['.','NULL'],
        ['He\'s','TARGET'],
        ['running','ACTION'],
        ['away from','MODIFIER'],
        ['you','GROUNDING'],
        ['!','NULL'],
        ['He\'s','TARGET'],
        ['behind','SPATIALRELATION'],
        ['the desk','GROUNDING'],
        [',','NULL'],
        ['about to','NULL'],
        ['leave','ACTION'],
        ['the kitchen','GROUNDING'],
        ['.','NULL'],
        ['Two robots','TARGET'],
        ['are','POSITIVITY'],
        ['moving','ACTION'],
        ['away from','MODIFIER'],
        ['each-other','GROUNDING'],
        ['.','NULL'],
        ['I','NULL'],
        ['think','NULL'],
        ['Pris','TARGET'],
        ['is','POSITIVITY'],
        ['trying to','NULL'],
        ['stay','ACTION'],
        ['in','SPATIALRELATION'],
        ['the kitchen','GROUNDING'],
        ['.','NULL'],
        ]
    return data

def generate_fleming_test_data():
    from cops_and_robots.human_tools.human import generate_human_language_template
    (certainties,
    positivities,
    relations,
    actions,
    modifiers,
    groundings,
    target_names) = generate_human_language_template()

    data = []
    for target in target_names:
        for positivity in positivities:

            # Spatial relation statements
            for grounding_type_name, grounding_type in groundings.iteritems():
                for grounding_name, grounding in grounding_type.iteritems():
                    grounding_name = grounding_name.lower()
                    if grounding_name == 'deckard':
                        continue
                    if grounding_name.find('the') == -1:
                        grounding_name = 'the ' + grounding_name
                    relation_names = grounding.relations.binary_models.keys()
                    for relation_name in relation_names:
                        relation = relation_name

                        # Make relation names grammatically correct
                        relation_name = relation_name.lower()
                        if relation_name == 'front':
                            relation_name = 'in front of'
                        elif relation_name == 'back':
                            relation_name = 'behind'
                        elif relation_name == 'left':
                            relation_name = 'left of'
                        elif relation_name == 'right':
                            relation_name = 'right of'

                        # Ignore certain questons
                        if grounding_type_name == 'object':
                            if relation_name in ['inside', 'outside']:
                                continue

                        # Write spatial relation tagged data
                        data.append(['I know', 'NULL'])
                        data.append([target, 'TARGET'])
                        data.append([positivity, 'POSITIVITY'])
                        data.append([relation_name, 'SPATIALRELATION'])
                        data.append([grounding_name, 'GROUNDING'])
                        data.append(['.', 'NULL'])

                    # Action statements
                    for action in actions:
                        if action == 'stopped':
                            data.append(['I know', 'NULL'])
                            data.append([target, 'TARGET'])
                            data.append([positivity, 'POSITIVITY'])
                            data.append([action, 'ACTION'])
                            data.append(['.', 'NULL'])
                            continue

                        for modifier in modifiers:

                            data.append(['I know', 'NULL'])
                            data.append([target, 'TARGET'])
                            data.append([positivity, 'POSITIVITY'])
                            data.append([action, 'ACTION'])
                            data.append([modifier, 'MODIFIER'])
                            if modifier in ['toward', 'around']:
                                data.append([grounding_name, 'GROUNDING'])

                                str_ = ("I know " + target + ' ' + positivity + ' '
                                        + action + ' ' + modifier + ' ' +
                                        grounding_name + '.')
                            else:
                                str_ = ("I know " + target + ' ' + positivity + ' '
                                        + action + ' ' + modifier + '.')
                            data.append(['.', 'NULL'])

    return data


if __name__ == '__main__':
    tagger = Tagger()
    print tagger.tag_document('I know Roy is behind the table')
