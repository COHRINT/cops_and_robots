#!/usr/bin/env python
import rospy
import sys, time, logging, getch
from cops_and_robots.Cop import Cop
from std_msgs.msg import String,Int8

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
    ser.write(cmd)
    tstep = 0.5
    time.sleep(tstep)

    
def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("robot_command", String, callback)
    rospy.spin()

if __name__ == '__main__':
    #Create Logger for console-level debugging
    logger = logging.getLogger('iRobot_interface')
    logger.addHandler(logging.StreamHandler()) #output to console
    logging.basicConfig(level=logging.DEBUG)

    cop = Cop()
    
    keymap = {  '.' : lambda: cop.faster(),
                ',' : lambda: cop.slower(),
                'w' : lambda: cop.forward(),
                's' : lambda: cop.backward(),
                'a' : lambda: cop.rotateCCW(),
                'd' : lambda: cop.rotateCW(),
                'q' : lambda: cop.turn(1000),
                'e' : lambda: cop.turn(-1000),
                ' ' : lambda: cop.stop() }

    listener()

    #allowing ctrl-c to close Cop thread (see http://www.regexprn.com/2010/05/killing-multithreaded-python-programs.html)
    while True:    
        try:
            cop.t.join(1)           
        except (KeyboardInterrupt, SystemExit):
            cop.thread_stop.set()
            break
