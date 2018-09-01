#!/usr/bin/env python
import rospy
from std_msgs.msg import String

import pyttsx

engine = pyttsx.init()


# subscribing to audio feedback
def audio_callback(audio_msg):
    # directly read it
    engine.say(audio_msg)
    engine.runAndWait()


if __name__ == "__main__":
    rospy.init_node("naive_experiment")

    # create a subscriber
    audio_sub = rospy.Subscriber("/audio_feedback", String, audio_callback)

    # loop
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
