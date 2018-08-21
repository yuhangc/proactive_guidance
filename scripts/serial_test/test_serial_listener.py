#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String


class Listener(object):
    def __init__(self):
        self.counter_received = 0

        self.haptic_msg_sub = rospy.Subscriber("/haptic_control", Float32MultiArray, self.received_callback)
        self.ctrl_pub = rospy.Publisher("/ctrl_received_listener", String, queue_size=1)

    def received_callback(self, msg):
        self.counter_received += 1

        pub_msg = String()
        pub_msg.data = str(self.counter_received)
        self.ctrl_pub.publish(pub_msg)

    def run(self, freq):
        rate = rospy.Rate(freq)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node("serial_test_listener")

    tester = Listener()
    tester.run(40)
