#!/usr/bin/env python
import numpy as np

import rospy
from std_msgs.msg import String
from people_msgs.msg import PositionMeasurementArray

from model_plan.gp_model_approx import GPModelApproxBase
from model_plan.movement_model import MovementModel
from model_plan.policies import NaivePolicy, MDPFixedTimePolicy


class PolicyExperiment(object):
    def __init__(self):
        # read in configurations
        
        pass

    def run(self, trial_start):
        pass


if __name__ == "__main__":
    pass
