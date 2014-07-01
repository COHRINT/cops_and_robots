#!/usr/bin/env python
import rospy
import sys, time, logging, getch
from cops_and_robots.Cop import Cop
from std_msgs.msg import String,Int8
from cops_and_robots.msg import battery


def callback(data):
    rospy.loginfo(rospy.get_caller_id()+"I heard %s",data.data)
    x = data.data

    print "4: AT LEAST IT GETS HERE!"
    
    try:
        keymap[x]()
    except Exception, e:
        logging.error('%s is not a viable command',x)
    
    
    logging.info('char: %s',x)
    logging.info('speed: %d mm/s',cop.speed)
    logging.info('radius: %d mm/s',cop.radius)

    cmd = cop.move()
    cop.cmd_queue.put(cmd)
    # tstep = 0.5
    # time.sleep(tstep)

def chatter():
    rospy.init_node('chatter', anonymous=True)

    print "3A: AT LEAST IT GETS HERE!"
    

    #listener
    rospy.Subscriber("robot_command", String, callback)
    # rospy.spin()

    print "3B: AT LEAST IT GETS HERE!"
    

    #talker
    pub = rospy.Publisher("battery",battery,queue_size=10)
    r = rospy.Rate(1) #1Hz
    while not rospy.is_shutdown():

        rospy.loginfo(cop.battery_charge)
        rospy.loginfo(cop.battery_capacity)
        pub.publish(cop.battery_capacity, cop.battery_charge, cop.charging_mode)
        r.sleep()

if __name__ == '__main__':
    #Create Logger for console-level debugging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler()) #output to console
    logging.basicConfig(level=logging.DEBUG)

    cop = Cop()
    cop.start_base_cx()
    
    keymap = {  '.' : lambda: cop.faster(),
                ',' : lambda: cop.slower(),
                'w' : lambda: cop.forward(),
                's' : lambda: cop.backward(),
                'a' : lambda: cop.rotateCCW(),
                'd' : lambda: cop.rotateCW(),
                'q' : lambda: cop.turn(800),
                'e' : lambda: cop.turn(-800),
                ' ' : lambda: cop.stop() }

    print "1: AT LEAST IT GETS HERE!"
    chatter()
    
    print "2: AT LEAST IT GETS HERE!"
    
    cop.stop_base_cx()
