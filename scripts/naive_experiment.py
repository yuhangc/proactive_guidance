#!/usr/bin/env python
import numpy as np

import rospy
from std_msgs.msg import Float32MultiArray
from people_msgs.msg import PositionMeasurementArray

import threading

from data_recorder import DataLogger
from utils import getKey


class NaiveExperiment:
    def __init__(self):
        # get protocol
        protocol_file = rospy.get_param("~protocol_file", "protocol.txt")

        self.t_trans = None
        self.dir = None
        self.mag = None

        self.load_protocol(protocol_file)

        # create a data logger
        path_saving = rospy.get_param("~path_saving", ".")
        self.logger = DataLogger(path_saving, False)

        self.flag_start_trial = False
        self.flag_end_trial = False
        self.flag_is_saving = False

        # create publisher
        self.haptic_msg_pub = rospy.Publisher("/haptic_control", Float32MultiArray, queue_size=1)

    def load_protocol(self, protocol_file):
        protocol_data = np.loadtxt(protocol_file, delimiter=", ")

        self.dir = protocol_data[:, 0]
        self.mag = protocol_data[:, 1]

    def _monitor_key(self):
        while not rospy.is_shutdown():
            key = getKey(0.1)

            if key == 's':
                self.flag_start_trial = True
            elif key == 'e':
                self.flag_end_trial = True

    def _loop(self, trial_start):
        trial = trial_start

        rate = rospy.Rate(40)
        while not rospy.is_shutdown():
            # save data if is saving
            if self.flag_is_saving:
                self.logger.log()

                # check for end trial
                if self.flag_end_trial:
                    self.logger.save_data(file_name="trial{}.txt".format(trial))
                    trial += 1

                    self.flag_is_saving = False
                    self.flag_end_trial = False
                    self.flag_start_trial = False
            else:
                # check for start trial
                if self.flag_start_trial:
                    # publish haptic feedback
                    haptic_msg = Float32MultiArray()
                    haptic_msg.data.append(self.dir[trial])
                    haptic_msg.data.append(self.mag[trial])

                    self.haptic_msg_pub.publish(haptic_msg)

                    # set flags
                    self.flag_is_saving = True
                    self.flag_start_trial = False
                    self.flag_end_trial = False

            rate.sleep()

    def run(self, trial_start):
        key_thread = threading.Thread(target=self._monitor_key).start()
        loop_thread = threading.Thread(target=self._loop, args=(trial_start)).start()
        key_thread.join()
        loop_thread.join()


if __name__ == "__main__":
    rospy.init_node("naive_experiment")

    exp = NaiveExperiment()
    exp.run(0)
