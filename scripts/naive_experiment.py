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
        self.mode = rospy.get_param("~mode", "manual")      # can be 'manual' or 'auto'
        self.t_render_inc = rospy.get_param("~t_render_inc", 1)
        self.t_render_offset = rospy.get_param("~t_render_offset", 5)
        self.t_pause = rospy.get_param("~t_pause", 3)

        self.t_trans = None
        self.dir = None
        self.mag = None

        self.mag_inc = 2.0
        self.mag_offset = 2.0
        self.pause_inc = 0.1
        self.pause_offset = 0.1

        self.load_protocol(protocol_file)

        # create a data logger
        path_saving = rospy.get_param("~path_saving", ".")
        self.logger = DataLogger(path_saving, False)

        self.t_start = 0.0
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
            elif key == '\x03':
                break

    def _loop(self, trial_start):
        trial = trial_start

        rate = rospy.Rate(40)

        t_last = rospy.get_time()
        t_render = 0
        while not rospy.is_shutdown():
            # save data if is saving
            if self.flag_is_saving:
                self.logger.log(self.t_start)

                # if mode is auto, set flag based on timer
                if self.mode == "auto" and rospy.get_time() - t_last > t_render:
                    self.flag_end_trial = True
                    t_last = rospy.get_time()

                # check for end trial
                if self.flag_end_trial:
                    self.logger.save_data(file_name="trial{}.txt".format(trial))
                    self.logger.reset()

                    self.flag_is_saving = False
                    self.flag_end_trial = False
                    self.flag_start_trial = False

                    # send another feedback to remind user
                    haptic_msg = Float32MultiArray()
                    haptic_msg.data.append(self.dir[trial])
                    haptic_msg.data.append(self.mag_offset)
                    haptic_msg.data.append(self.pause_offset)

                    self.haptic_msg_pub.publish(haptic_msg)

                    print "Trial ", trial, " ended\r"
                    trial += 1
            else:
                if self.mode == "auto" and rospy.get_time() - t_last > self.t_pause:
                    self.flag_start_trial = True
                    t_last = rospy.get_time()

                # check for start trial
                if self.flag_start_trial:
                    # publish haptic feedback
                    haptic_msg = Float32MultiArray()
                    haptic_msg.data.append(self.dir[trial])
                    haptic_msg.data.append(self.mag[trial] * self.mag_inc + self.mag_offset)
                    haptic_msg.data.append(self.mag[trial] * self.pause_inc + self.pause_offset)

                    self.haptic_msg_pub.publish(haptic_msg)

                    # set flags
                    self.flag_is_saving = True
                    self.flag_start_trial = False
                    self.flag_end_trial = False

                    t_render = self.mag[trial] * self.t_render_inc + self.t_render_offset

                    print "Trial ", trial, " started...\r"

            rate.sleep()

    def run(self, trial_start):
        self.t_start = rospy.get_time()

        if self.mode == "manual":
            key_thread = threading.Thread(target=self._monitor_key)
            loop_thread = threading.Thread(target=self._loop, args=[trial_start])
            key_thread.start()
            loop_thread.start()

            print "Please press 's' to start a trial, and 'e' to end a trial.\r"

            key_thread.join()
            loop_thread.join()
        else:
            print "Trial will start automatically in 3 seconds\r"
            self._loop(trial_start)


if __name__ == "__main__":
    rospy.init_node("naive_experiment")

    exp = NaiveExperiment()
    exp.run(0)
