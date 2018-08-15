#!/usr/bin/env python
import rospy
from std_msgs.msg import String

import numpy as np
from utils import getKey

msg_init = """
Send command to the haptic device
---------------------------
Send a direction with:
   u    i    o
   j    k    l
   m    ,    .
   
w - increase magnitude
s - decrease magnitude
a - decrease pause
d - increase pause

CTRL-C to quit
"""


ctrl_keys = ['u', 'i', 'o', 'j', 'k', 'l', 'm', ',', '.']
adj_keys = ['w', 's', 'a', 'd']

mag = 4.0       # mm
pause = 0.2     # s
dmag = 1.0
dpause = 0.2
mag_range = (2.0, 8.0)
pause_range = (0.0, 1.0)


if __name__ == "__main__":
    rospy.init_node('haptic_ctrl')

    pub = rospy.Publisher("haptic_control", String)

    print msg_init

    while True:
        key = getKey()
        if key in ctrl_keys:
            msg = String()
            msg.data = key
            pub.publish(msg)
        elif key in adj_keys:
            if key == 'w':
                mag += dmag
            elif key == 's':
                mag -= dmag
            elif key == 'a':
                pause -= dpause
            elif key == 'd':
                pause += dpause

            # clip mag and pause
            mag = np.clip(mag, mag_range[0], mag_range[1])
            pause = np.clip(pause, pause_range[0], pause_range[1])

            # print out information
            print "magnitude is: ", mag, "mm,  pause is: ", pause, "s"

            # publish
            msg = String()
            msg.data = key
            pub.publish(msg)
        else:
            if key == '\x03':
                break
