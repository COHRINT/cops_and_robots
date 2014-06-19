#!/usr/bin/env python
# license removed for brevity
import rospy, time
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('robot_command', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    r = rospy.Rate(10) # 10hz
    
    cmds = ['.','w','w','.','w','s',' ']
    for cmd in cmds:
        rospy.loginfo(cmd)
        pub.publish(cmd)
        time.sleep(1)
        r.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass
