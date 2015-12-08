from __future__ import division
#!/usr/bin/env python

import logging
import random
import time
import numpy as np
import itertools

class Questioner(object):

    def __init__(self, human_sensor=None, target_order=None, target_weights=1,
                 use_ROS=False, repeat_annoyance=0.5, repeat_time_penalty=60,
                 auto_answer=False, sequence_length=1, ask_every_n=0,
                 minimize_questions=False):
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
        return '\n'.join(self.all_questions)

    def remove_target(self, target):
        if self.target_order == []:
            logging.info('All targets found! No need to ask any more questions.')
            return
        self.target_order.remove(target)

        delete_indices = []
        for i, _ in enumerate(self.all_likelihoods):
            if target in self.all_likelihoods[i]['question']:
                delete_indices.append(i)
        self.all_likelihoods = np.delete(self.all_likelihoods, delete_indices, axis=0)
        self.all_questions = [q for i, q in enumerate(self.all_questions)\
                              if i not in delete_indices]

    def generate_questions(self):
        if self.human_sensor is None:
            certainties =['know if']
            targets =['a robot', 'nothing',] + target_order
            positivities = ['is']
            groundings = {'object':{
                                   'Deckard': ['in front of', 'behind', 'left of', 'right of', 'near'],
                                   'the chair': ['in front of', 'behind', 'left of', 'right of'],
                                   'the table':['beside', 'near', 'at the head of'],
                                   'the desk': ['in front of', 'behind', 'left of', 'right of'],
                                   'the filer': ['in front of', 'left of', 'right of'],
                                   'the bookcase': ['in front of', 'left of', 'right of'],
                                   'the billiard poster': ['in front of', 'left of', 'right of'],
                                   'the kitchen poster': ['in front of', 'left of', 'right of'],
                                   'the fridge': ['in front of', 'left of', 'right of'],
                                   'the checkers': ['in front of', 'left of', 'right of', 'behind'],
                         },
                         'area':{
                                  'the hallway': ['inside','outside','near'],
                                  'the kitchen': ['inside','outside','near'],
                                  'the billiard room': ['inside','outside','near'],
                                  'the study': ['inside','outside','near'],
                                  'the library': ['inside','outside','near'],
                                  'the dining room': ['inside','outside','near'],
                         }
                        }
        else:
            certainties = self.human_sensor.certainties
            targets = self.target_order
            positivities = self.human_sensor.positivities
            groundings = self.human_sensor.groundings

        # Create all possible questions and precompute their likelihoods
        n = len(certainties) * len(targets)
        if self.minimize_questions:
            num_relations = 1  # Only asking about 'near'
        else:
            num_relations = len(groundings['object'].itervalues().next()\
                    .relations.binary_models.keys())
            num_relations -= 1  # not counting 'inside' for objects
        num_questions = (len(groundings['object']) - 1) * num_relations * n

        if self.minimize_questions:
            num_relations = 1  # Only asking about inside rooms
        else:
            num_relations = len(groundings['area'].itervalues().next()\
                    .relations.binary_models.keys())
        num_questions += len(groundings['area']) * num_relations * n
        self.all_questions = []
        self.all_likelihoods = np.empty(num_questions,
                                        dtype=[('question', np.object),
                                               ('probability', np.object),
                                               ('time_last_answered', np.float)
                                               ])
        i = 0
        #<>TODO: include certainties
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

                    # Ignore certain questons (incl. non-minimal)
                    if grounding_type_name == 'object':
                        if relation_name in ['inside', 'outside']:
                            continue

                        if self.minimize_questions:
                            # Near only
                            if relation_name != 'near':
                                continue

                    if grounding_type_name == 'area' \
                        and self.minimize_questions: 
                        
                        # Inside only
                        if relation_name != 'inside':
                            continue

                    for target in targets:
                        # Write question
                        question_str = "Is " + target + " " + relation_name \
                            + " " + grounding_name +"?"
                        self.all_questions.append(question_str)

                        # Calculate likelihood
                        self.all_likelihoods[i]['question'] = \
                            question_str
                        self.all_likelihoods[i]['probability'] = \
                            grounding.relations.probability(class_=relation)
                        self.all_likelihoods[i]['time_last_answered'] = -1
                        i += 1
        logging.info('Generated {} questions.'.format(len(self.all_questions)))

    # def weigh_questions(self, priors):
    #     """Orders questions by their value of information.
    #     """
    #     # Calculate VOI for each question
    #     q_weights = np.empty_like(self.all_questions, dtype=np.float64)
    #     VOIs = np.empty_like(self.all_questions, dtype=np.float64)
    #     for prior_name, prior in priors.iteritems():
    #         prior_entropy = prior.entropy()
    #         flat_prior_pdf = prior.as_grid().flatten()

    #         for i, likelihood_obj in enumerate(self.all_likelihoods):
                
    #             # Check if this question concerns the correct target
    #             question = likelihood_obj['question']
    #             if prior_name.lower() not in question.lower():
    #                 continue

    #             # Use positive and negative answers for VOI
    #             likelihood = likelihood_obj['probability']
    #             VOIs[i] = self._calculate_VOI(likelihood, flat_prior_pdf,
    #                                                prior_entropy)
    #             q_weights[i] = VOIs[i]

    #             # Add heuristic question cost based on target weight
    #             for j, target in enumerate(self.target_order):
    #                 if target.lower() in question.lower():
    #                     q_weights[i] *= self.target_weights[j]

    #             # Add heuristic question cost based on number of times asked
    #             tla = likelihood_obj['time_last_answered']
    #             if tla == -1:
    #                 continue

    #             dt = time.time() - tla
    #             if dt > self.repeat_time_penalty:
    #                 self.all_likelihoods[i]['time_last_answered'] = -1
    #             elif tla > 0:
    #                 q_weights[i] *= (dt / self.repeat_time_penalty + 1)\
    #                      * self.repeat_annoyance

    #     # Re-order questions by their weights
    #     q_ids = range(len(self.all_questions))
    #     self.weighted_questions = zip(q_weights, q_ids, self.all_questions[:])
    #     self.weighted_questions.sort(reverse=True)
    #     self.VOIs = VOIs
    #     for q in self.weighted_questions:
    #         logging.debug(q)

    # def _calculate_VOI(self, likelihood, flat_prior_pdf, prior_entropy=None):
    #     """Calculates the value of a specific question's information.

    #     VOI is defined as:

    #     .. math::

    #         VOI(i) = \\sum_{j \\in {0,1}} P(D_i = j)
    #             \\left(-\\int p(x \\vert D_i=j) \\log{p(x \\vert D_i=j)}dx\\right)
    #             +\\int p(x) \\log{p(x)}dx

    #     Takes VOI of a specific branch.
    #     """
    #     if prior_entropy is None:
    #         prior_entropy = prior.entropy()

    #     VOI = 0
    #     grid_spacing = 0.1 #prior.res

    #     alpha = self.human_sensor.false_alarm_prob / 2  # only for binary
    #     pos_likelihood = alpha + (1 - alpha) * likelihood
    #     neg_likelihood = np.ones_like(pos_likelihood) - pos_likelihood
    #     neg_likelihood = alpha + (1 - alpha) * neg_likelihood
    #     likelihoods = [neg_likelihood, pos_likelihood]

    #     # flat_prior_pdf = prior.as_grid().flatten()
    #     for likelihood in likelihoods:
    #         post_unnormalized = likelihood * flat_prior_pdf
    #         sensor_marginal = np.sum(post_unnormalized) * grid_spacing ** 2
    #         log_post_unnormalized = np.log(post_unnormalized)
    #         log_sensor_marginal = np.log(sensor_marginal)

    #         VOI += -np.nansum(post_unnormalized * (log_post_unnormalized - 
    #             log_sensor_marginal)) * grid_spacing ** 2

    #     VOI += -prior_entropy
    #     return -VOI  # keep value positive

    def weigh_questions(self, priors):
        """Orders questions by their value of information.
        """
        # Calculate VOI for each question
        question_sequences = list(itertools.product(self.all_questions,
                                                    repeat=self.sequence_length
                                                    ))
        likelihood_sequences = list(itertools.product(self.all_likelihoods,
                                                      repeat=self.sequence_length
                                                      ))

        # Figure out the timespan considered
        K = (self.sequence_length - 1) * self.ask_every_n

        VOIs = np.empty(len(likelihood_sequences),
                        dtype=np.float64)
        question_sequence_weights = np.empty(len(likelihood_sequences),
                                             dtype=np.float64)
        for prior_name, prior in priors.iteritems():
            posterior = prior.copy()  # we may need to modify the prior
            initial_probability = prior.copy()  # we may need to modify the prior

            #Find the entropy and PDF without any questions at K
            posterior.dynamics_update(n_steps=K)
            posterior_entropy = posterior.entropy()
            # flat_posterior_pdf = posterior.as_grid().flatten()

            for s, likelihood_sequence in enumerate(likelihood_sequences):

                # # Check if this question concerns the correct target
                # concerns_target = True
                # for likelihood_obj in likelihood_objs:
                #     question = likelihood_obj['question']
                #     if prior_name.lower() not in question.lower():
                #         concerns_target = False
                # if not concerns_target:
                #     break

                # Use positive and negative answers for VOI
                likelihood_seq_values = []
                for likelihood_seq in likelihood_sequence:
                    likelihood_seq_values.append(likelihood_seq['probability'])

                VOIs[s] = self._calculate_VOI(likelihood_seq_values, 
                                              prior=prior,
                                              probability=initial_probability,
                                              final_posterior_entropy=posterior_entropy,
                                              timespan=K
                                              )
                question_sequence_weights[s] = VOIs[s]

                # Add heuristic question cost based on target weight
                for j, target in enumerate(self.target_order):
                    for question in question_sequences[s]:
                        if target.lower() in question.lower():
                            question_sequence_weights[s] *= self.target_weights[j]

                # Add heuristic question cost based on number of times asked
                tla = likelihood_sequence[0]['time_last_answered']
                if tla == -1:
                    continue

                dt = time.time() - tla
                qid = self.all_questions.index(likelihood_sequence[0]['question'])
                if dt > self.repeat_time_penalty:
                    self.all_likelihoods[qid]['time_last_answered'] = -1
                elif tla > 0:
                    question_sequence_weights[s] *= (dt / self.repeat_time_penalty + 1)\
                         * self.repeat_annoyance

        # Re-order questions by their weights
        q_seq_ids = range(len(likelihood_sequences))

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

                qid = self.all_questions.index(wqs[2][0])
                weighted_question = [wqs[0], qid, wqs[2][0]]
                weighted_questions.append(weighted_question)

                question_weights.append(wqs[0])
                question_strs.append(wqs[2][0])
        self.weighted_questions = weighted_questions


        DEBUG = False
        if DEBUG:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            width = 0.8
            # ax.bar(range(len(question_sequences)), question_sequence_weights, width=width)
            # ax.set_xticks(np.arange(len(question_sequences)) + width/2)
            # question_strs = [" + ".join(question_seq) for question_seq in question_sequences]
            # ax.set_xticklabels(question_strs, rotation=90)

            ax.bar(range(len(question_strs)), question_weights, width=width)
            ax.set_xticks(np.arange(len(question_strs)) + width/2)
            ax.set_xticklabels(question_strs, rotation=90, fontsize=10)
            fig.subplots_adjust(bottom=0.4)

            plt.show()

        for q in self.weighted_questions:
            logging.debug(q)

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

                # Perform a Bayes' update
                posterior = likelihood * probability.prob.flatten()
                data_likelihood *= posterior.sum()
                posterior /= posterior.sum()
                probability.prob = np.reshape(posterior, probability.X.shape)

                # Perform dynamics update
                if timespan  > 0 and d < (len(answer_sequence) - 1)\
                    and probability.is_dynamic:

                    probability.dynamics_update()

            sequence_entropy[s] = probability.entropy() * data_likelihood

        average_sequence_entropy = sequence_entropy.sum()

        VOI = final_posterior_entropy - average_sequence_entropy
        return VOI

    def ask(self, priors, i=0, num_questions_to_ask=5, ask_human=True,
            robot_positions=None):

        # Don't ask if it's not time
        if (self.ask_every_n < 1):
            return

        if (i % self.ask_every_n) != (self.ask_every_n - 1):
            self.human_sensor.question_str = ''
            self.human_sensor.utterance = ''
            return

        if self.is_hovered and self.use_ROS:
            return

        # Get questions sorted by weight
        self.weigh_questions(priors)
        questions_to_ask = self.weighted_questions[:num_questions_to_ask]

        if self.auto_answer:
            self.answer_question(self.weighted_questions[0], robot_positions)
            self.recent_question = self.weighted_questions[0][2]
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
            self.all_likelihoods[q[1]]['time_last_answered'] = time.time()

    def answer_question(self, question, robot_positions):
        qid = question[1]
        question_str = question[2]

        # assume target is Roy
        true_position = robot_positions['Roy']

        groundings = self.human_sensor.groundings
        positivities = self.human_sensor.positivities
        map_bounds = None
        answer = None
        for _, grounding_type in groundings.iteritems():

            for grounding_name, grounding in grounding_type.iteritems():
                if grounding_name.find('the') == -1:
                    grounding_name = 'the ' + grounding_name

                if hasattr(grounding, 'relations') and map_bounds is None:
                    map_bounds =  grounding.relations.binary_models\
                        .itervalues().next().bounds

                if not hasattr(grounding, 'relations'):
                    grounding.define_relations()

                relation_names = grounding.relations.binary_models.keys()

                for relation in relation_names:

                    if grounding_name.lower() in question_str \
                        and relation.lower() in question_str:

                        p_D_x = grounding.relations.probability(class_=relation,
                                                                state=true_position
                                                                )
                        sample = np.random.random()
                        if sample < p_D_x:
                            answer = True
                        else:
                            answer = False
                        break
                else:
                    continue
                break
            else:
                continue
            break
        if answer is None:
            logging.error('No answer available to the question "{}"!'
                          .format(question_str))
        else:
            if answer:
                self.recent_answer = 'Yes'
            else:
                self.recent_answer = 'No'
            logging.info('{} {}'.format(question_str, self.recent_answer))

        self.question_str = " + ".join(self.weighted_question_sequences[0][2])
        self.human_sensor.question_str = question_str
        statement = self.question_to_statement(question_str, answer)
        self.human_sensor.utterance = statement
        self.human_sensor.new_update = True
        self.all_likelihoods[qid]['time_last_answered'] = time.time()


    def hover_callback(self, data):
        import rospy
        self.is_hovered = data.data
        rospy.loginfo("%s", data.data)

    def answer_callback(self, data):
        import rospy
        index = data.qid
        answer = data.answer
        logging.info("I Heard{}".format(data.qid))

        question = self.all_questions[index]
        self.statement = self.question_to_statement(question, answer)
        logging.info(self.statement)

        self.answer_publisher.publish(self.statement)
        self.all_likelihoods[index]['time_last_answered']

    def question_to_statement(self, question, answer):
        statement = question.replace("Is", "I know")
        statement = statement.replace("?", ".")
        for target in self.target_order:
            start = statement.find(target)
            if start > -1:
                l = len(target)
                i = start + l
                statement = statement[:i] + ' is' + statement[i:]

        if answer == 0 or answer == 'n' or answer == False:
            statement = statement.replace("is", "is not")

        return statement


def test_publishing():
    rospy.init_node('question', anonymous=True)
    questioner = Questioner()
    while not rospy.is_shutdown():
        questioner.ask()
        questioner.rate.sleep()


def test_voi():
    from cops_and_robots.robo_tools.robber import Robber
    from cops_and_robots.map_tools.map import Map
    from cops_and_robots.fusion.human import Human
    from cops_and_robots.fusion.gaussian_mixture import GaussianMixture
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
    m.add_human_sensor(h)

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
    prior._discretize(bounds=m.bounds, grid_spacing=0.1)
    q = Questioner(human_sensor=h, target_order=['Pris','Roy'],
                   target_weights=[11., 10.])

    m.setup_plot(show_human_interface=False)
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
