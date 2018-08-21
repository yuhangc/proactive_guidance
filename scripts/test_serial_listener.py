#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String


class Listener(object):
    def __init__(self):
        self.counter_received = 0

        self.haptic_msg_pub = rospy.Publisher("/haptic_control", Float32MultiArray, queue_size=1)
        self.ctrl_sub = rospy.Subscriber("/ctrl_received_listener", String, self.received_callback)

    def received_callback(self, msg):
        self.counter_received += 1

        pub_msg = String()
        pub_msg.data = str(self.counter_received)

    def run(self, freq):
        rate = rospy.Rate(freq)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node("serial_test_listener")

    tester = Listener()
    tester.run(40)
