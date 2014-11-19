#!/usr/bin/env python
import rospy
import sys, time, logging
from geometry_msgs.msg import PoseStamped
from cops_and_robots.Map import Map

goal = [0,0]

def callback(data):
    rospy.loginfo(rospy.get_caller_id()+"I heard %s",data.data)
    str = data.data

    target = str.split()[2]
    obj = str.split()[-2:]
    obj = ''.join(obj)[:-1].capitalize()

    fleming.probability[target].update(fleming.objects[obj],'front')
    goal = fleming.probability[target].ML

def nav_goals():
    rospy.init_node('nav_goals', anonymous=True)

    #listener
    rospy.Subscriber("human_sensor", String, callback)    

    #talker
    pub = rospy.Publisher("nav_goal", PoseStamped, queue_size=10)
    pose = PoseStamped()

    r = rospy.Rate(1) #1Hz
    while not rospy.is_shutdown():
        pose.header.frame_id = 'base_link'
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = goal[0]
        pose.pose.position.y = goal[1]
        pose.pose.position.z = 0    
        orientation = tf.transformations.quaternion_from_euler(0, 0, 0)
        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]

        pub.publish(pose)
        r.sleep()

if __name__ == '__main__':
    try:
        nav_goals()
    except rospy.ROSInterruptException: pass

    