#!/usr/bin/env python
import numpy as np

import rospy
from std_msgs.msg import Float32MultiArray
from people_msgs.msg import PositionMeasurementArray


class PolicyBase(object):
    def __init__(self):
        pass

