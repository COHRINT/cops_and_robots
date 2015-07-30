from __future__ import division
#!/usr/bin/env python

import logging
import random
import numpy as np

class Questioner(object):

    def __init__(self, human_sensor=None, use_ROS=False, target_order=None):
        self.human_sensor = human_sensor
        self.use_ROS = use_ROS
        self.target_order = target_order
        self.update_target()
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

    def update_target(self, target=None):
        if self.target_order == []:
            logging.info('All targets found! No need to ask any more questions.')
            return
        if target==None or target==self.target:
            self.target = self.target_order.pop(0)
        else:
            self.target_order.remove(target)

        logging.info('Asking about {}.'.format(self.target))
        self.generate_questions()

    def generate_questions(self):
        if self.human_sensor is None:
            certainties =['know if']
            targets =['a robot', 'nothing', 'Roy', 'Pris', 'Zhora']
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
            if self.target is None:
                targets = self.human_sensor.target_names
            else:
                targets = [self.target]
            positivities = self.human_sensor.positivities
            groundings = self.human_sensor.groundings

        # Create all possible questions and precompute their likelihoods
        n = len(certainties) * len(targets)
        num_questions = len(groundings['object']) * 4 * n
        num_questions += len(groundings['area']) * 3 * n
        self.all_questions = []
        self.all_likelihoods = np.empty(num_questions,
                                        dtype=[('question', np.object),
                                               ('probability', np.object)
                                               ])
        i = 0
        for grounding_type_name, grounding_type in groundings.iteritems():
            for grounding_name, grounding in grounding_type.iteritems():
                grounding_name = grounding_name.lower()
                if grounding_name.find('the') == -1:
                    grounding_name = 'the ' + grounding_name
                relation_names = grounding.relations.binary_models.keys()
                for relation_name in relation_names:
                    relation = relation_name
                    relation_name = relation_name.lower()
                    if relation_name == 'front':
                        relation_name = 'in front of'
                    elif relation_name == 'back':
                        relation_name = 'behind'
                    elif relation_name == 'left':
                        relation_name = 'left of'
                    elif relation_name == 'right':
                        relation_name = 'right of'
                    if grounding_type_name == 'object' and relation_name == 'inside':
                        continue
                    for certainty in certainties:
                        if certainty == 'know':
                            certainty = 'know if'
                        for target in targets:
                            # Write question
                            question_str = "Do you " + certainty + " " + \
                                target + " is " + relation_name + " " + \
                                grounding_name +"?"
                            self.all_questions.append(question_str)

                            # Calculate likelihood
                            self.all_likelihoods[i]['question'] = \
                                question_str
                            self.all_likelihoods[i]['probability'] = \
                                grounding.relations.probability(class_=relation)
                            i += 1

    def weigh_questions(self, prior):
        """Orders questions by their value of information.
        """
        # Calculate VOI for each question
        q_weights = np.empty_like(self.all_questions, dtype=np.float64)
        prior_entropy = prior.entropy()
        flat_prior_pdf = prior.pdf().flatten()

        for i, question in enumerate(self.all_likelihoods):
            # Use positive and negative answers for VOI
            likelihood = self.all_likelihoods[i]
            q_weights[i] = self._calculate_VOI(likelihood, flat_prior_pdf,
                                               prior_entropy)

        # Re-order questions by their weights
        q_ids = range(len(self.all_questions))
        self.weighted_questions = zip(q_weights, q_ids, self.all_questions[:])
        self.weighted_questions.sort(reverse=True)
        

    def _calculate_VOI(self, likelihood, flat_prior_pdf, prior_entropy=None):
        """Calculates the value of a specific question's information.

        VOI is defined as:

        .. math::

            VOI(i) = \\sum_{j \\in {0,1}} P(D_i = j)
                \\left(-\\int p(x \\vert D_i=j) \\log{p(x \\vert D_i=j)}dx\\right)
                +\\int p(x) \\log{p(x)}dx
        """
        if prior_entropy is None:
            prior_entropy = prior.entropy()

        VOI = 0
        grid_spacing = 0.1
        alpha = self.human_sensor.false_alarm_prob / 2  # only for binary
        pos_likelihood = alpha + (1 - alpha) * likelihood['probability']
        neg_likelihood = np.ones_like(pos_likelihood) - pos_likelihood
        neg_likelihood = alpha + (1 - alpha) * neg_likelihood
        likelihoods = [neg_likelihood, pos_likelihood]
        for likelihood in likelihoods:
            post_unnormalized = likelihood * flat_prior_pdf
            sensor_marginal = np.sum(post_unnormalized) * grid_spacing ** 2
            log_post_unnormalized = np.log(post_unnormalized)
            log_sensor_marginal = np.log(sensor_marginal)

            VOI += -np.sum(post_unnormalized * (log_post_unnormalized - 
                log_sensor_marginal)) * grid_spacing ** 2

        VOI += -prior_entropy
        return -VOI  # keep value positive

    def ask(self, prior, num_questions_to_ask=5, ask_human=True):
        if self.is_hovered and self.use_ROS:
            pass
        else:
            # Get questions sorted by weight
            self.weigh_questions(prior)
            questions_to_ask = self.weighted_questions[:num_questions_to_ask]

            # Assign values to the ROS message
            if self.use_ROS:
                self.message.weights = []
                self.message.qids = []
                self.message.questions = []
                
                normalizer = 0
                for q in questions_to_ask:
                    normalizer += q[0]
                # normalizer = np.sum(questions_to_ask[:num_questions_to_ask][0])
                for q in questions_to_ask:
                    normalized_weight = q[0] / normalizer
                    self.message.weights.append(normalized_weight)
                    self.message.qids.append(q[1])
                    self.message.questions.append(q[2])

                logging.debug(self.message)
                self.question_publisher.publish(self.message)
            elif ask_human:
                qu = self.weighted_questions[0]
                logging.info(qu[2])
                answer = raw_input('Answer (y/n): ')
                statement = question_to_statement(qu[2], answer)
                self.human_sensor.utterance = statement
                self.human_sensor.new_update = True


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
        self.statement = question_to_statement(question, answer)
        logging.info(self.statement)
        self.answer_publisher.publish(self.statement)

def question_to_statement(question, answer):
    statement = question.replace("Do you", "I")
    statement = statement.replace("?", ".")
    statement = statement.replace("know if", "know")
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
    q = Questioner(human_sensor=h, target='Roy')

    m.setup_plot(show_human_interface=False)
    m.update()
    ax = m.axes['combined']

    prior.plot(bounds=m.bounds, alpha=0.5, ax=ax)
    m.update()
    plt.show()
    q.weigh_questions(prior)
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