#!/usr/bin/env python
import rospy
from std_msgs.msg import String

import sys, select, termios, tty

msg_init = """
Send command to the haptic device
---------------------------
Send a direction with:
   u    i    o
   j    k    l
   m    ,    .
CTRL-C to quit
"""


ctrl_keys = ['u', 'i', 'o', 'j', 'k', 'l', 'm', ',', '.']


def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


if __name__=="__main__":
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('haptic_ctrl')

    pub = rospy.Publisher("haptic_control", String)

    print msg_init

    while True:
        key = getKey()
        if key in ctrl_keys:
            msg = String()
            msg.data = key
            pub.publish(msg)
        else:
            if key == '\x03':
                break
