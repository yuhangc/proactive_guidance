#!/usr/bin/env python
import rospy
from std_msgs.msg import String

import pyttsx

engine = pyttsx.init()


# subscribing to audio feedback
def audio_callback(audio_msg):
    # split string to get direction and mag
    if audio_msg.data == "270,0":
        engine.say("stop")
    else:
        cmd = audio_msg.data.split(',')
        
        direction = int(cmd[0]) - 90
        if direction > 180:
            direction -= 360
        
        if direction > 0:
            engine.say("left")
            engine.say(str(direction))
        else:
            engine.say("right")
            engine.say(str(-direction))


if __name__ == "__main__":
    rospy.init_node("naive_experiment")

    # create a subscriber
    audio_sub = rospy.Subscriber("/audio_feedback", String, audio_callback)

    # loop
    rate = rospy.Rate(40)
    engine.startLoop(False)
    while not rospy.is_shutdown():
        engine.iterate()
        rate.sleep()

    engine.endLoop()

