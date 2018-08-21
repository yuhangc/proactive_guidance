#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String


class Tester(object):
    def __init__(self):
        self.counter_sent = 0
        self.counter_received = 0
        self.counter_listener = 0

        self.haptic_msg_pub = rospy.Publisher("/haptic_control", Float32MultiArray, queue_size=1)
        self.ctrl_sub = rospy.Subscriber("/ctrl_received", String, self.received_callback)
        self.ctrl_sub2 = rospy.Subscriber("/ctrl_received_listener", String, self.lintener_callback)

    def received_callback(self, msg):
        self.counter_received += 1

    def lintener_callback(self, msg):
        self.counter_listener += 1

    def run(self, freq):
        rate = rospy.Rate(freq)
        while not rospy.is_shutdown():
            msg = Float32MultiArray()
            msg.data.append(0)
            msg.data.append(0)
            msg.data.append(0)

            self.haptic_msg_pub.publish(msg)
            self.counter_sent += 1

            rate.sleep()

            # check for received
            if self.counter_sent % freq == 0:
                print self.counter_sent, "messages sent,", self.counter_received, \
                    "messages received, success rate is", float(self.counter_received) / self.counter_sent
                print "\t", self.counter_listener, "messages received from listener, success rate is", \
                    float(self.counter_listener) / self.counter_sent


if __name__ == '__main__':
    rospy.init_node("serial_test")

    tester = Tester()
    tester.run(2)
