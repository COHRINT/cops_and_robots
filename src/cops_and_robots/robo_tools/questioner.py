from __future__ import division
#!/usr/bin/env python

import logging
import random
import time
import numpy as np
import itertools
import pickle
import os
from sklearn.gaussian_process import GaussianProcess

from cops_and_robots.human_tools.statement_template import get_all_statements


class Questioner(object):

    def __init__(self, human_sensor=None, target_order=None, target_weights=1,
                 use_ROS=False, repeat_annoyance=0.5, repeat_time_penalty=60,
                 auto_answer=False, sequence_length=1, ask_every_n=0,
                 minimize_questions=False, mask_file='area_masks.npy',
                 GP_VOI_file='', use_GP_VOI=False):
        self.human_sensor = human_sensor
        self.use_ROS = use_ROS
        self.target_order = target_order
        self.repeat_annoyance = repeat_annoyance
        self.repeat_time_penalty = repeat_time_penalty
        self.ask_every_n = ask_every_n
        self.sequence_length = sequence_length
        self.auto_answer = auto_answer
        self.minimize_questions = minimize_questions

        # Set up target order and 
        target_weights = np.asarray(target_weights, dtype=np.float32)
        if target_weights.size == len(target_order):
            target_weights /= target_weights.sum()
            self.target_weights = target_weights
        else:
            logging.warn("Target weights don't match target order. Assuming "
                "equal weights.")
            self.target_weights = np.ones(len(target_order)) * 1 / len(target_order)
        self.generate_questions()
        self.is_hovered = False

        self.use_GP_VOI = use_GP_VOI
        if self.use_GP_VOI:
            filepath = os.path.dirname(os.path.abspath(__file__)) \
                + '/VOI-GPs/' + GP_VOI_file
            fh = open(filepath, 'r')
            self.GPs = pickle.load(fh)

            # Set up areas
            mask_file = os.path.dirname(os.path.abspath(__file__)) \
                + '/VOI-GPs/' + mask_file
            self.area_masks = np.load(mask_file).reshape(-1)[0]  #Black magic
            self.area_probs = {'Billiard Room': [],
                       'Hallway': [],
                       'Study': [],
                       'Library': [],
                       'Dining Room': [],
                       'Kitchen': []
                      }

        if self.use_ROS:
            import rospy
            from std_msgs.msg import Bool
            from std_msgs.msg import String
            from cops_and_robots_ros.msg import Question, Answer

            self.message = Question()  # allocate now, init later

            self.question_publisher = rospy.Publisher('questions',
                                                      Question,
                                                      queue_size=10)
            self.answer_publisher = rospy.Publisher('human_sensor',
                                                    String,
                                                    queue_size=10)
            self.hover_subscriber = rospy.Subscriber('question_hover',
                                                     Bool,
                                                     self.hover_callback)
            self.answer_subscriber = rospy.Subscriber('answer',
                                                      Answer,
                                                      self.answer_callback)
            self.rate = rospy.Rate(.2)  # frequency of question updates [Hz]

    def __str__(self):
        return '\n'.join(self.questions)

    def remove_target(self, target):
        if self.target_order == []:
            logging.info('All targets found! No need to ask any more questions.')
            return
        self.target_order.remove(target)

        delete_indices = []
        for i, _ in enumerate(self.likelihoods):
            if target in self.likelihoods[i]['question']:
                delete_indices.append(i)
        self.likelihoods = np.delete(self.likelihoods, delete_indices, axis=0)
        self.questions = [q for i, q in enumerate(self.questions)\
                              if i not in delete_indices]

    def generate_questions(self):
        """Generates all questions, pre-computing likelihood functions.
        """

        # Define questions from all positive statements
        self.statements = get_all_statements(autogenerate_softmax=True,
                                             flatten=True)
        self.questions = [s.get_question_string() for s in self.statements
                          if s.positivity == 'is' and not s.target == 'nothing']

        # Map questioner's statements to only positive statements
        self.statements = [s for s in self.statements
                           if s.positivity == 'is' and not s.target == 'nothing']

        # Pre-Generate likelihoods for all questions/statements
        self.likelihoods = np.empty(len(self.questions),
                                        dtype=[('question', np.object),
                                               ('probability', np.object),
                                               ('time_last_answered', np.float),
                                               ])
        for i, question in enumerate(self.questions):
            self.likelihoods[i]['question'] = question
            lh = self.statements[i].get_likelihood(discretized=True)
            self.likelihoods[i]['probability'] = lh
            self.likelihoods[i]['time_last_answered'] = -1
        logging.info('Generated {} questions.'.format(len(self.questions)))


    def weigh_questions(self, priors):
        """Orders questions by their value of information.

        We use question *sequences* for non-myopic questioning. Myopic
        questioning is simply the case in which the sequence length is 1. VOI
        is calculated for each possible sequence as a whole, using the 
        expected entropy of all possible combinations of answers down a branch
        of questions.
        """
        # List out the possible combinations of all questions
        question_sequences = list(itertools.product(self.questions,
                                                    repeat=self.sequence_length
                                                    ))
        likelihood_sequences = list(itertools.product(self.likelihoods,
                                                      repeat=self.sequence_length
                                                      ))

        # Figure out the timespan considered
        K = (self.sequence_length - 1) * self.ask_every_n

        VOIs = np.empty(len(likelihood_sequences),
                        dtype=np.float64)
        question_sequence_weights = np.empty(len(likelihood_sequences),
                                             dtype=np.float64)

        # Assuming one prior belief per target, calculate VOI for that target
        for target_name, prior in priors.iteritems():
            posterior = prior.copy()  # we may need to modify the prior
            initial_probability = prior.copy()

            #Find the entropy and PDF without any questions at K
            posterior.dynamics_update(n_steps=K)
            posterior_entropy = posterior.entropy()
            # flat_posterior_pdf = posterior.as_grid().flatten()

            # Go through possible question paths
            for s, likelihood_sequence in enumerate(likelihood_sequences):

                # # Check if this question concerns the correct target
                # concerns_target = True
                # for likelihood_obj in likelihood_objs:
                #     question = likelihood_obj['question']
                #     if target_name.lower() not in question.lower():
                #         concerns_target = False
                # if not concerns_target:
                #     break

                # Use positive and negative answers for VOI
                likelihood_seq_values = [ls['probability'] for ls in likelihood_sequence]

                # Ignore questions that have no associated sensor likelihoods
                if any([ls is None for ls in likelihood_seq_values]):
                    VOIs[s] = np.nan
                    question_sequence_weights[s] = -np.inf
                    continue

                # Approximate or use brute-force VOI
                if self.use_GP_VOI:
                    self._get_area_probs(prior)
                    VOIs[s] = self._predict_VOI(s)
                else:
                    VOIs[s] = self._calculate_VOI(likelihood_seq_values, 
                                                  prior=prior,
                                                  probability=initial_probability,
                                                  final_posterior_entropy=posterior_entropy,
                                                  timespan=K
                                                  )

                # Ensure positive weights
                question_sequence_weights[s] = VOIs[s] - np.nanmin(VOIs[~np.isneginf(VOIs)])

                # Add heuristic question cost based on target weight
                for j, target in enumerate(self.target_order):
                    for question in question_sequences[s]:
                        if target.lower() in question.lower():
                            question_sequence_weights[s] *= self.target_weights[j]

                # Add heuristic question cost based on time last answered
                tla = likelihood_sequence[0]['time_last_answered']
                if np.isnan(tla):
                    continue

                dt = time.time() - tla
                qid = self.questions.index(likelihood_sequence[0]['question'])
                if dt > self.repeat_time_penalty:
                    self.likelihoods[qid]['time_last_answered'] = np.nan
                elif tla > 0:
                    mp = 1 - self.repeat_annoyance
                    question_sequence_weights[s] *= mp + (1-mp) * dt / self.repeat_time_penalty

        # Re-order questions by their weights
        q_seq_ids = range(len(likelihood_sequences))

        # <>TODO: change to NamedTuple
        self.weighted_question_sequences = zip(question_sequence_weights, 
                                               q_seq_ids,
                                               question_sequences,
                                               )
        self.weighted_question_sequences.sort(reverse=True)

        self.VOIs = VOIs
        self.ordered_question_list = [" + ".join(q) for q in question_sequences]


        # Filter down to the weighted first questions in each path
        weighted_questions = []
        first_questions = []
        question_weights = []
        question_strs = []
        qid = 0
        for wqs in self.weighted_question_sequences:

            if wqs[2][0] not in first_questions:
                first_questions.append(wqs[2][0])

                qid = self.questions.index(wqs[2][0])
                weighted_question = [wqs[0], qid, wqs[2][0]]
                weighted_questions.append(weighted_question)

                question_weights.append(wqs[0])
                question_strs.append(wqs[2][0])
        self.weighted_questions = weighted_questions

    def _calculate_VOI(self, likelihood_seq_values, prior, probability=None,
                       final_posterior_entropy=None, timespan=0,
                      ):
        """Calculates the value of a specific question's information.

        VOI is defined as:

        .. math::

            VOI(i) = \\sum_{j \\in {0,1}} P(D_i = j)
                \\left(-\\int p(x \\vert D_i=j) \\log{p(x \\vert D_i=j)}dx\\right)
                +\\int p(x) \\log{p(x)}dx

        Takes VOI of a specific branch.
        """
        if probability is None:
            probability = prior.copy()

        alpha = self.human_sensor.false_alarm_prob / 2  # only for binary
        answer_sequences = list(itertools.product([False, True],
                                                  repeat=self.sequence_length))
        sequence_entropy = np.empty(len(answer_sequences))

        # Go through the whole answer tree
        for s, answer_sequence in enumerate(answer_sequences):
            probability.prob = prior.prob
            data_likelihood = 1

            # Go through one answer tree branch
            for d, answer in enumerate(answer_sequence):
                pos_likelihood = likelihood_seq_values[d]

                # Get likelihood based on answer (with human error)
                likelihood = alpha + (1 - alpha) * pos_likelihood
                if not answer:
                    likelihood = np.ones_like(likelihood) - likelihood

                # Perform a Bayes' update on the discretized probability
                posterior = likelihood * probability.prob.flatten()
                data_likelihood *= posterior.sum()
                posterior /= posterior.sum()
                probability.prob = np.reshape(posterior, prior.prob.shape)

                # Perform dynamics update
                if timespan  > 0 and d < (len(answer_sequence) - 1)\
                    and probability.is_dynamic:

                    probability.dynamics_update()

            sequence_entropy[s] = probability.entropy() * data_likelihood

        average_sequence_entropy = sequence_entropy.sum()

        VOI = final_posterior_entropy - average_sequence_entropy
        return VOI

    def _predict_VOI(self, s):
        question = self.likelihoods[s][0]
        VOI, MSE = self.GPs[question].predict(self.eval_points, eval_MSE=True)
        sigma = np.sqrt(MSE)

        return VOI

    def _get_area_probs(self, prior):

        # Get each area's CDF for each timestep
        p = prior.prob.flatten()
        for area_name, area_mask in self.area_masks.iteritems():
            area_prob = (p*area_mask).sum()
            self.area_probs[area_name].append(area_prob)

        ndim_input = 6
        self.eval_points = np.zeros(ndim_input)
        i = 0
        for _, area_prob in self.area_probs.iteritems():
            self.eval_points[i] = area_prob[0]
            i += 1

    def ask(self, priors, i=0, num_questions_to_ask=5, ask_human=True,
            robot_positions=None):

        # Don't ask if it's not time
        if ((self.ask_every_n < 1) or  # Too early
            (i % self.ask_every_n) != (self.ask_every_n - 1) or  # Wrong timestep
            (self.is_hovered and self.use_ROS)):  # user hovering
            return

        # Get questions sorted by weight
        self.weigh_questions(priors)
        questions_to_ask = self.weighted_questions[:num_questions_to_ask]

        if self.auto_answer:
            self.answer_question(self.weighted_questions[0], robot_positions)
            return

        # Assign values to the ROS message
        if self.use_ROS:
            self.message.weights = []
            self.message.qids = []
            self.message.questions = []
            
            # Normalize questions to show some differences in the output
            normalizer = 0
            for q in questions_to_ask:
                normalizer += q[0]

            # Create the question messages
            for q in questions_to_ask:
                normalized_weight = q[0] / normalizer
                self.message.weights.append(normalized_weight)
                self.message.qids.append(q[1])
                self.message.questions.append(q[2])

            # Publish the question messages
            logging.debug(self.message)
            self.question_publisher.publish(self.message)
        elif ask_human:
            # Get human to answer a single question
            q = self.weighted_questions[0]
            logging.info(q[2])
            answer = raw_input('Answer (y/n): ')
            statement = self.question_to_statement(q[2], answer)
            self.human_sensor.utterance = statement
            self.human_sensor.new_update = True
            self.likelihoods[q[1]]['time_last_answered'] = time.time()

    def answer_question(self, question, robot_positions):

        # Find the exact statement tied to the question
        qid = question[1]
        question_str = question[2]
        statement = self.statements[qid]

        # Find the position of the robot
        if statement.target in ['a robot', 'nothing']:
            positions = [p for _, p in robot_positions.iteritems()]
            true_position = np.array(positions).mean(axis=0)
        else:
            true_position = robot_positions[statement.target]

        # Randomly sample one answer from the likelihood at target location
        #<>TODO: consider the case of dynamic groundings
        likelihood = statement.get_likelihood(state=true_position)
        answer = np.random.random() < likelihood
        answer_statement = self.question_to_statement(question, answer)

        # Update relevant values
        self.question_str = " + ".join(self.weighted_question_sequences[0][2])
        self.human_sensor.add_new_measurement(answer_statement)
        self.likelihoods[qid]['time_last_answered'] = time.time()
        self.recent_answer = 'Yes' if answer else 'No'
        self.recent_question = self.weighted_questions[0][2]
        logging.info('{} {}'.format(question_str, self.recent_answer))

    def question_to_statement(self, question, answer):
        """Maps a question and an answer to a template statement.

        Returns the statement if the answer was affirmative, or finds the
        negative statement and returns that one if the answer was negative.
        """
        qid = question[1]
        pos_statement = self.statements[qid]
        if answer:
            return pos_statement
        else:
            utterance = pos_statement.__str__().replace('is', 'is not')
            return self.human_sensor.utterance_to_statement(utterance)

    def hover_callback(self, data):
        import rospy
        self.is_hovered = data.data
        rospy.loginfo("%s", data.data)

    def answer_callback(self, data):
        import rospy
        index = data.qid
        answer = data.answer
        logging.info("I Heard{}".format(data.qid))

        question = self.questions[index]
        self.statement = self.question_to_statement(question, answer)
        logging.info(self.statement)

        self.answer_publisher.publish(self.statement)
        self.likelihoods[index]['time_last_answered']


def test_publishing():
    rospy.init_node('question', anonymous=True)
    questioner = Questioner()
    while not rospy.is_shutdown():
        questioner.ask()
        questioner.rate.sleep()


def test_voi():
    from cops_and_robots.robo_tools.robber import Robber
    from cops_and_robots.map_tools.map import Map
    from cops_and_robots.fusion.gaussian_mixture import GaussianMixture
    from cops_and_robots.human_tools.human import Human

    import matplotlib.pyplot as plt
    from matplotlib.colors import cnames

    m = Map()
    pris = Robber('Pris')
    pris.map_obj.color = cnames['cornflowerblue']
    m.add_robber(pris.map_obj)
    zhora = Robber('Zhora')
    zhora.map_obj.color = cnames['cornflowerblue']
    m.add_robber(zhora.map_obj)
    roy = Robber('Roy')
    m.add_robber(roy.map_obj)
    h = Human(map_=m)
    # m.add_human_sensor(h)

    prior = GaussianMixture([0.1, 0.7, 0.2],
                            [[3, -2],
                             [-6, -2],
                             [0, 0]
                             ], 
                             [[[1.5, 1.0],
                               [1.0, 1.5]],
                              [[2.5, -0.3],
                               [-0.3, 2.5]],
                              [[0.5, -0.3],
                               [-0.3, 0.5]]
                             ])
    prior._discretize(bounds=m.bounds, res=0.1)
    q = Questioner(human_sensor=h, target_order=['Pris','Roy'],
                   target_weights=[11., 10.])

    m.setup_plot()
    m.update()
    ax = m.axes['combined']

    # prior.plot(bounds=m.bounds, alpha=0.5, ax=ax)
    # m.update()
    # plt.show()
    q.weigh_questions({'Roy':prior})
    for qu in q.weighted_questions:
        print qu
    # loop_length = 10
    # for i in range(loop_length):
    #     plt.cla()
    #     m.update()
    #     prior.plot(bounds=m.bounds, alpha=0.5, ax=ax)

    #     q.weigh_questions(prior)
    #     print q.weighted_questions[0]
    #     answer = raw_input('Answer: ')
    #     if answer == 'y':
    #         prior = h.detect()
    #     else:
    #         prior = h.detect

if __name__ == '__main__':
    # test_publishing()
    test_voi()
