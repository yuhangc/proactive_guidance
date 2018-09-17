#!/usr/bin/env python
import numpy as np

import rospy
from std_msgs.msg import String
from people_msgs.msg import PositionMeasurementArray

import threading

from data_recorder import DataLogger
from utils import getKey, wrap_to_pi


class NaiveExperimentBase(object):
    def __init__(self):
        # get protocol
        protocol_file = rospy.get_param("~protocol_file", "protocol.txt")
        self.mode = rospy.get_param("~mode", "manual")      # can be 'manual' or 'auto'

        self.t_trans = None
        self.dir = None
        self.mag = None

        self.load_protocol(protocol_file)

        # create a data logger
        path_saving = rospy.get_param("~path_saving", ".")
        self.logger = DataLogger(path_saving, False)
        
        self.robot_pose = rospy.get_param("~robot_pose", [0.0, 1.0, np.pi * 0.5])
        self.x_range = rospy.get_param("~x_range", [-5.0, 5.0])
        self.y_range = rospy.get_param("~y_range", [0.5, 7.0])
        self.logger.load_env_config(self.robot_pose, [self.x_range, self.y_range])

        self.t_start = 0.0
        self.flag_start_trial = False
        self.flag_end_trial = False
        self.flag_is_saving = False

        self.flag_end_program = False

        # create publisher
        self.haptic_msg_pub = rospy.Publisher("/haptic_control", String, queue_size=1)

    def load_protocol(self, protocol_file):
        protocol_data = np.loadtxt(protocol_file, delimiter=", ")

        self.dir = protocol_data[:, 0]
        self.mag = protocol_data[:, 1]

    def publish_haptic_control(self, ctrl):
        raise Exception("Method must be implemented!")

    def _monitor_key(self):
        while not rospy.is_shutdown():
            key = getKey(0.1)

            if key == 's':
                self.flag_start_trial = True
            elif key == 'e':
                self.flag_end_trial = True
            elif key == '\x03':
                break

        self.flag_end_program = True
        print "Experiment terminated..."

    def _loop(self, trial_start):
        raise Exception("Method must be implemented!")

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
            print "Trial will start automatically in 5 seconds\r"
            self._loop(trial_start)


class NaiveExperimentDiscreteCue(NaiveExperimentBase):
    def __init__(self):
        super(NaiveExperimentDiscreteCue, self).__init__()

        self.t_render_inc = rospy.get_param("~t_render_inc", 1)
        self.t_render_offset = rospy.get_param("~t_render_offset", 5)
        self.t_pause = rospy.get_param("~t_pause", 3)

    def publish_haptic_control(self, ctrl):
        # publish haptic feedback
        haptic_msg = String()
        haptic_msg.data = "{:d}{:d}{:d}".format(int(ctrl[0]),
                                                int(ctrl[1]),
                                                int(ctrl[2]))
        print haptic_msg.data

        self.haptic_msg_pub.publish(haptic_msg)

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

                    self.flag_is_saving = False
                    self.flag_end_trial = False
                    self.flag_start_trial = False

                    # send another feedback to remind user
                    self.publish_haptic_control([self.dir[trial], 0])

                    print "Trial ", trial, " ended\r"
                    trial += 1
            else:
                if self.mode == "auto" and rospy.get_time() - t_last > self.t_pause:
                    self.flag_start_trial = True
                    t_last = rospy.get_time()

                # check for start trial
                if self.flag_start_trial:
                    # publish haptic feedback
                    self.publish_haptic_control([self.dir[trial], self.mag[trial]])

                    # set flags
                    self.flag_is_saving = True
                    self.flag_start_trial = False
                    self.flag_end_trial = False

                    t_render = self.mag[trial] * self.t_render_inc + self.t_render_offset

                    # reset logger
                    self.logger.reset()

                    print "Trial ", trial, " started...\r"

            rate.sleep()


class NaiveExperimentContinuousCue(NaiveExperimentBase):
    def __init__(self):
        super(NaiveExperimentContinuousCue, self).__init__()

        self.t_render = rospy.get_param("~t_render", 5)
        self.t_pause = rospy.get_param("~t_pause", 5)
        self.n_block = rospy.get_param("~n_block", 40)

    def publish_haptic_control(self, ctrl):
        # publish haptic feedback
        haptic_msg = String()
        haptic_msg.data = "{:d},{:d}".format(int(ctrl[0]), int(ctrl[1]))
        print haptic_msg.data

        self.haptic_msg_pub.publish(haptic_msg)

    def _break(self, rate):
        print "Now in break, please press 's' to start next block...\r"
        while not rospy.is_shutdown():
            if self.flag_start_trial:
                break
            rate.sleep()

        self.flag_start_trial = False
        print "Break ends, starting trial in ", self.t_pause, "seconds\r"

    def _loop(self, trial_start):
        trial = trial_start

        rate = rospy.Rate(40)

        # first wait for calibration
        print "Please calibrate the IMU, when ready, press 's'...\r"

        while not rospy.is_shutdown():
            print "Calibration values are: ", self.logger.cal_data, "\r"
            if self.flag_start_trial:
                break
            rate.sleep()

        self.flag_start_trial = False

        print "Calibration finished...\r"
        if self.mode == "auto":
            print "Trial will automatically start in ", self.t_pause, " seconds\r"
        else:
            print "Please press 's' to start trial and 'e' to end trial\r"

        t_last = rospy.get_time()
        while not rospy.is_shutdown() and not self.flag_end_program:
            # save data if is saving
            if self.flag_is_saving:
                self.logger.log(self.t_start)

                # if mode is auto, set flag based on timer
                if self.mode == "auto" and rospy.get_time() - t_last > self.t_render:
                    self.flag_end_trial = True
                    t_last = rospy.get_time()

                # check for end trial
                if self.flag_end_trial:
                    self.logger.save_data(file_name="trial{}.txt".format(trial))

                    self.flag_is_saving = False
                    self.flag_end_trial = False
                    self.flag_start_trial = False

                    # send another feedback to remind user
                    # always use backward cue for this
                    self.publish_haptic_control([270, 0])

                    print "Trial ", trial, " ended\r"
                    trial += 1

                    # take a break or end experiment if reaching end of the block
                    if trial >= len(self.dir):
                        break

                    if trial % self.n_block == 0:
                        self._break(rate)

                    t_last = rospy.get_time()
            else:
                if self.mode == "auto" and rospy.get_time() - t_last > self.t_pause:
                    self.flag_start_trial = True
                    t_last = rospy.get_time()

                # check for start trial
                if self.flag_start_trial:
                    # publish haptic feedback
                    self.publish_haptic_control([self.dir[trial], self.mag[trial]])

                    # set flags
                    self.flag_is_saving = True
                    self.flag_start_trial = False
                    self.flag_end_trial = False

                    # reset logger
                    self.logger.reset()

                    print "Trial ", trial, " started...\r"
                    t_last = rospy.get_time()

            rate.sleep()

    def run(self, trial_start):
        self.t_start = rospy.get_time()

        key_thread = threading.Thread(target=self._monitor_key)
        loop_thread = threading.Thread(target=self._loop, args=[trial_start])
        key_thread.start()
        loop_thread.start()

        key_thread.join()
        loop_thread.join()


class NaiveExperimentContinuousGuide(NaiveExperimentContinuousCue):
    def __init__(self):
        super(NaiveExperimentContinuousGuide, self).__init__()

        self.t_render = 4.0
        self.t_pause = 6.0

        self.t_render_inc = 2.0 / np.pi

    def _loop(self, trial_start):
        trial = trial_start

        rate = rospy.Rate(40)

        # first wait for calibration
        print "Please calibrate the IMU, when ready, press 's'...\r"

        while not rospy.is_shutdown():
            print "Calibration values are: ", self.logger.cal_data, "\r"
            if self.flag_start_trial:
                break
            rate.sleep()

        self.flag_start_trial = False

        print "Calibration finished...\r"
        if self.mode == "auto":
            print "Trial will automatically start in ", self.t_pause, " seconds\r"
        else:
            print "Please press 's' to start trial and 'e' to end trial\r"

        t_last = rospy.get_time()
        t_render = self.t_render
        while not rospy.is_shutdown() and not self.flag_end_program:
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

                    self.flag_is_saving = False
                    self.flag_end_trial = False
                    self.flag_start_trial = False

                    # send another feedback to remind user
                    # always use backward cue for this
                    self.publish_haptic_control([270, 0])

                    print "Trial ", trial, " ended\r"
                    trial += 1

                    # take a break or end experiment if reaching end of the block
                    if trial >= len(self.dir):
                        break

                    if trial % self.n_block == 0:
                        self._break(rate)

                    t_last = rospy.get_time()
            else:
                if self.mode == "auto" and rospy.get_time() - t_last > self.t_pause:
                    self.flag_start_trial = True
                    t_last = rospy.get_time()

                # check for start trial
                if self.flag_start_trial:
                    # publish haptic feedback
                    self.publish_haptic_control([self.dir[trial], self.mag[trial]])

                    # set flags
                    self.flag_is_saving = True
                    self.flag_start_trial = False
                    self.flag_end_trial = False

                    # reset logger
                    self.logger.reset()

                    # compute new render time
                    t_render = self.t_render + \
                               np.abs(wrap_to_pi(np.deg2rad(self.dir[trial]) - np.pi * 0.5)) * self.t_render_inc

                    print "Trial ", trial, " started, t_render is ", t_render, "...\r"
                    t_last = rospy.get_time()

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("naive_experiment")

    # exp = NaiveExperimentDiscreteCue()
    # exp = NaiveExperimentContinuousCue()
    exp = NaiveExperimentContinuousGuide()
    exp.run(0)
