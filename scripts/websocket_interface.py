#!/usr/bin/env python
import rospy
import sys, time, logging, getch
from cops_and_robots.Cop import Cop
from std_msgs.msg import String,Int8
# from .. import battery


def callback(data):
    rospy.loginfo(rospy.get_caller_id()+"I heard %s",data.data)
    x = data.data

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

    #listener
    rospy.Subscriber("robot_command", String, callback)
    rospy.spin()

    #talker
    pub = rospy.Publisher("battery",Int8,queue_size=10)
    r = rospy.Rate(1) #1Hz
    while not rospy.is_shutdown():
        
        if cop.battery_capacity > 0:
            pct = cop.battery_charge/cop.battery_capacity
        else:
            pct = 0
        rospy.loginfo(pct)
        pub.publish(pct)
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

    chatter()
    

    # cop.stop_base_cx()
