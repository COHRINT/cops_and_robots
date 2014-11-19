#!/usr/bin/env python
import rospy
import sys, time, logging
import tf
import math
from geometry_msgs.msg import PoseStamped
from cops_and_robots.Map import *
from std_msgs.msg import String

fleming = set_up_fleming()
goal = [0,0]

def callback(data):
    global goal
    rospy.loginfo(rospy.get_caller_id()+"I heard %s",data.data)
    str_ = data.data

    target = str_.split()[2]
    obj = str_.split()[-2:]
    obj = ''.join(obj)[:-1].capitalize()

    fleming.probability[target].update(fleming.objects[obj],'front')
    goal = fleming.probability[target].ML
    rospy.loginfo('New Goal' + str(goal))
    return goal

def nav_goals():
    global goal
    rospy.init_node('nav_goals', anonymous=True)

    #listener
    rospy.Subscriber("human_sensor", String, callback)    

    #talker
    pub = rospy.Publisher("nav_goal", PoseStamped, queue_size=10)
    ps = PoseStamped()

    r = rospy.Rate(1) #1Hz
    while not rospy.is_shutdown():
	rospy.loginfo('Pose Goal' + str(goal))
        ps.header.frame_id = 'base_link'
        ps.header.stamp = rospy.Time.now()
        try:
            ps.pose.position.x = goal[0]
            ps.pose.position.y = goal[1]
        except:
            goal = [0,0]
            ps.pose.position.x = 0
            ps.pose.position.y = 0
        ps.pose.position.z = 0    
        orientation = tf.transformations.quaternion_from_euler(0, 0, 0)
        ps.pose.orientation.x = orientation[0]
        ps.pose.orientation.y = orientation[1]
        ps.pose.orientation.z = orientation[2]
        ps.pose.orientation.w = orientation[3]

        pub.publish(ps)
        r.sleep()

if __name__ == '__main__':
    try:
        nav_goals()
    except rospy.ROSInterruptException: pass

    
