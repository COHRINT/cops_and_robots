#!/usr/bin/env python

import rospy
import random
import numpy as np
from cops_and_robots_ros.msg import Question
from cops_and_robots_ros.msg import question_answer
from std_msgs.msg import Bool
from std_msgs.msg import String

class Questions(object):

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
    all_questions=[]
    for certinty in certainties:
        for target in targets:
            for positive in positivities:
                for dict in groundings.values():
                    for element,relations in dict.iteritems():
                        for relation in relations:
                            all_questions.append("Do you "+certinty+" "+target+" "+positive+" "+relation+" "+element+"?")
    # for question in all_questions:
    #     print question
    # print len(all_questions)

    def __init__(self):
        self.pub = rospy.Publisher('questions', Question, queue_size=10)
        self.pub1 = rospy.Publisher("human_sensor", String, queue_size=10)
        self.sub = rospy.Subscriber("question_hover", Bool, self.callback)
        self.sub1 = rospy.Subscriber('question_answer', question_answer, self.statement)
        rospy.init_node('question', anonymous=True)
        self.rate = rospy.Rate(.2)
        self.all_questions = Questions.all_questions
        self.message = Question()
        self.is_hovered = False

    def ask(self):
        if self.is_hovered:
            pass
        else:
            self.message.questions = random.sample(self.all_questions,5)
            self.message.weights = np.random.uniform(0, 1, 5)
            self.message.qids = []
            self.message.weights /= self.message.weights.sum()
            joined=zip(self.message.weights, self.message.questions)
            joined.sort(reverse=True)
            self.message.weights = zip(*joined)[0]
            self.message.questions = zip(*joined)[1]
            for i in self.message.questions:
                self.message.qids.append(self.all_questions.index(i))
            rospy.loginfo(self.message)
            self.pub.publish(self.message)

    def callback(self, data):
        self.is_hovered = data.data
        rospy.loginfo("%s", data.data)

    def statement(self, data):
        self.index = data.qid
        self.answer = data.answer
        rospy.loginfo("%s", data.qid)
        asked = self.all_questions[self.index]
        self.statement = asked.replace("Do you", "I")
        self.statement = self.statement.replace("?", ".")
        self.statement = self.statement.replace("know if", "know")
        if not self.answer:
            self.statement = self.statement.replace("is", "is not")
        rospy.loginfo(self.statement)
        self.pub1.publish(self.statement)


if __name__ == '__main__':
    question = Questions()
    while not rospy.is_shutdown():
        question.ask()
        question.rate.sleep()
    