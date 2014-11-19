import rospy
import sys, time, logging, getch
from cops_and_robots.Map import Map
from std_msgs.msg import String

#<>NOTE: THIS WILL NOT WORK IN PYTHON AND HAS TO BE WRITTEN IN C++
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>



def callback(data):
    rospy.loginfo(rospy.get_caller_id()+"I heard %s",data.data)
    x = data.data
    
    try:
        keymap[x]()
    except Exception, e:
        logging.error('%s is not a viable command',x)

    cmd = cop.move()
    cop.cmd_queue.put(cmd)

def chatter():
    rospy.init_node('map_chatter', anonymous=True)

    #listener
    rospy.Subscriber("human_sensor", String, callback)    

    #talker
    pub = rospy.Publisher("battery", battery, queue_size=10)
    r = rospy.Rate(1) #1Hz
    while not rospy.is_shutdown():
        pub.publish(cop.battery_capacity, cop.battery_charge, cop.charging_mode)
        r.sleep()


if __name__ == '__main__':
	fleming = set_up_fleming()

	chatter()