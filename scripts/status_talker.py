#!/usr/bin/env python
import rospy
import sys, time, logging, getch
from cops_and_robots.Cop import Cop
from std_msgs.msg import Int8

def talker():
    pub = rospy.Publisher('battery',Int8,queue_size=10)
    rospy.init_node('talker', anonymous=True)
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
    logger = logging.getLogger('moveTest')
    logger.addHandler(logging.StreamHandler()) #output to console
    logging.basicConfig(level=logging.DEBUG)

    cop = Cop('Deckard')
    OPCODE = cop.OPCODE
    ser = cop.ser

    ser.write(chr(OPCODE['start']) + chr(OPCODE['full']))

    talker()
