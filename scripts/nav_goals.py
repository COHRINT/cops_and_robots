#!/usr/bin/env python
import roslib; #roslib.load_manifest('cops_and_robots')
import rospy
import actionlib
import sys, time, logging
import tf
import math
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist
from cops_and_robots.Map import *
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class MoveBase():
    def __init__(self):
        self.map_ = set_up_fleming()
        self.goal2d = [0,0]

        rospy.init_node('nav_goals',anonymous=True)
        rospy.on_shutdown(self.shutdown)

        #Start subscriber to update goal2d from MAP
        rospy.Subscriber("/human_sensor", String, self.callback)    

        self.move_base = actionlib.SimpleActionClient('move_base',MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base.wait_for_server(rospy.Duration.from_sec(5.0))
        rospy.loginfo("Connected to move_base action server.")

        # Goal state return values
        goal_states = ['PENDING', 'ACTIVE', 'PREEMPTED', 'SUCCEEDED', 'ABORTED', 'REJECTED', 'PREEMPTING', 'RECALLING', 'RECALLED', 'LOST']

        self.goal = MoveBaseGoal()
        while not rospy.is_shutdown():
            rospy.loginfo('Pose Goal ' + str(self.goal2d))
            try:
                self.goal.target_pose.pose.position.x = self.goal2d[0]
                self.goal.target_pose.pose.position.y = self.goal2d[1]
            except:
                self.goal2d = [0,0]
                self.goal.target_pose.pose.position.x = 0
                self.goal.target_pose.pose.position.y = 0
            self.goal.target_pose.pose.orientation = Quaternion(0, 0, 0, 1)
            self.goal.target_pose.header.frame_id = 'map'
            self.goal.target_pose.header.stamp = rospy.Time.now()
            self.move()

    def move(self):
        # Start the robot toward the next location
        self.move_base.send_goal(self.goal)

        # Allow 1 minute to get there
        finished_within_time = self.move_base.wait_for_result(rospy.Duration(60)) 

        if not finished_within_time:
            self.move_base.cancel_goal()
            rospy.loginfo("Timed out achieving goal")
        else:
            state = self.move_base.get_state()
            # if state == GoalStatus.SUCCEEDED:
            #     rospy.loginfo("Goal succeeded!")

    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        self.move_base.cancel_goal()
        rospy.sleep(2)
        # self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)
    
    def callback(self,data):
        rospy.loginfo(rospy.get_caller_id()+"I heard %s",data.data)
        str_ = data.data

        target = str_.split()[2]
        obj = str_.split()[-2:]
        obj = '_'.join(obj)[:-1].capitalize()

        self.map_.probability[target].update(self.map_.objects[obj],'front')
        self.goal2d = self.map_.probability[target].ML
        rospy.loginfo('New Goal ' + str(self.goal2d))

if __name__ == '__main__':
    print('Test1')
    try:
        MoveBase()
    except rospy.ROSInterruptException: pass

    
