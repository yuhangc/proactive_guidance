#!/usr/bin/env python
import numpy as np

import rospy
from std_msgs.msg import String
from people_msgs.msg import PositionMeasurementArray

import threading

from data_recorder import DataLogger
from utils import getKey, wrap_to_pi


class RandomGuidanceExp(object):
    def __init__(self):
        # read in configurations
        # feedback modality
        self.modality = rospy.get_param("~modality", "haptic")

        # time interval to update plan
        self.planner_dt = rospy.get_param("~planner_dt", 2.0)

        # time interval for resetting/pausing
        self.resetting_dt = rospy.get_param("~resetting_dt", 3.0)
        self.resuming_dt = rospy.get_param("~resuming_dt", 5.0)
        self.trial_dt = rospy.get_param("~trial_dt", 5.0)
        self.t_reset_start = 0.0
        self.t_resume_start = 0.0
        self.t_trial_start = 0.0

        # goal reaching threshold
        self.goal_reaching_th = rospy.get_param("~goal_reaching_th", 0.5)

        # create a data logger
        path_saving = rospy.get_param("~path_saving", ".")
        self.robot_pose = rospy.get_param("~robot_pose", [0.0, 1.0, np.pi * 0.5])
        self.x_range = rospy.get_param("~x_range", [-5.0, 5.0])
        self.y_range = rospy.get_param("~y_range", [-1.0, 6.0])

        self.logger = DataLogger(path_saving, False)
        self.logger.load_env_config(self.robot_pose, [self.x_range, self.y_range])

        # experiment mode
        self.mode = rospy.get_param("~mode", "manual")      # can be 'manual' or 'auto'

        # load protocol
        self.n_trial_total = 60
        self.trial = 0

        # goal that is randomly sampled each time
        self.s_g = np.zeros((3, ))

        # a stop command
        self.msg_stop = [270, 0]

        # controlling the behavior of the manager
        self.t_start = 0.0
        self.flag_start_trial = False
        self.flag_end_trial = False
        self.flag_pause = False
        self.flag_is_saving = False

        self.flag_end_program = False

        # valid states are "Running", "Pausing", "Resetting", "Resuming"
        self.state = "Pausing"

        # create publisher
        self.haptic_msg_pub = rospy.Publisher("/haptic_control", String, queue_size=1)

    def publish_haptic_control(self, ctrl):
        # publish haptic feedback
        haptic_msg = String()
        haptic_msg.data = "{:d},{:d}".format(int(ctrl[0]), int(ctrl[1]))
        print haptic_msg.data, "\r"

        self.haptic_msg_pub.publish(haptic_msg)

    def _monitor_key(self):
        while not rospy.is_shutdown():
            key = getKey(0.1)

            if key == 's':
                self.flag_start_trial = True
            elif key == 'e':
                self.flag_end_trial = True
            elif key == 'p':
                self.flag_pause = True
            elif key == '\x03':
                break

        self.flag_end_program = True
        print "Experiment terminated...\r"

    def check_for_stop(self, s):
        err = np.linalg.norm(s[:2] - self.s_g)

        if err < self.goal_reaching_th:
            # send a stop command
            print "Goal reached!\r"
            self.publish_haptic_control(self.msg_stop)

            return True
        else:
            return False

    def convert_feedback(self, cmd):
        # first from body-centered to device-centered
        cmd += np.pi * 0.5

        # convert to deg and 0-360
        cmd = np.rad2deg(cmd)
        if cmd < 0:
            cmd += 360

        # round to integer based on modality
        if self.modality == "haptic":
            cmd = int(np.round(cmd))
        elif self.modality == "audio":
            cmd = int(np.round(cmd / 5.0)) * 5

        return cmd

    def check_goal_pos(self, x, y):
        # check angle to robot
        ang = np.arctan2(y - self.robot_pose[1], x - self.robot_pose[0])
        if ang >= np.pi * 0.75 or ang <= -np.pi * 0.75:
            return False
        else:
            return True

    def start_trial(self):
        self.flag_start_trial = False

        # randomly sample a goal
        x = np.random.uniform(self.x_range[0], self.x_range[1])
        y = np.random.uniform(self.y_range[0], self.y_range[1])

        while not self.check_goal_pos(x, y):
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            y = np.random.uniform(self.y_range[0], self.y_range[1])

        self.s_g = np.array([x, y, 0.0])

        # reset data logger
        self.logger.reset()

        print "Starting trial ", self.trial, "...\r"
        self.t_trial_start = rospy.get_time()

        self.state = "Running"

    def _loop(self, trial_start):
        self.trial = trial_start

        rate = rospy.Rate(40)

        # first wait for calibration to complete
        print "Please calibrate the IMU, when ready, press 's'...\r"

        while not rospy.is_shutdown():
            print "Calibration values are: ", self.logger.cal_data, "\r"
            if self.flag_start_trial:
                break
            rate.sleep()

        print "Experiment started...\r"

        self.state = "Resetting"
        self.flag_start_trial = False
        t_last = rospy.get_time()
        self.t_reset_start = rospy.get_time()

        while not rospy.is_shutdown() and not self.flag_end_program:
            if self.state == "Running":
                # check for stop or trial end
                pose = self.logger.get_pose()

                if self.check_for_stop(pose) or rospy.get_time() - self.t_trial_start > self.trial_dt:
                    print "Trial ", self.trial, " ended\r"

                    # save data first
                    self.logger.save_data(file_name="trial{}".format(self.trial), flag_save_comm=True)

                    # check if exp ends
                    self.trial += 1
                    if self.trial >= self.n_trial_total:
                        print "All trials finished!\r"
                        break

                    # prepare to reset
                    self.t_reset_start = rospy.get_time()
                    self.state = "Resetting"

                if rospy.get_time() - t_last >= self.planner_dt:
                    # compute and send feedback
                    x_diff = self.s_g[:2] - pose[:2]
                    cmd = wrap_to_pi(np.arctan2(x_diff[1], x_diff[0]) - pose[2])

                    # convert to right format and publish
                    self.publish_haptic_control([self.convert_feedback(cmd), 2])

                    # log communication
                    self.logger.log_comm(rospy.get_time() - self.t_trial_start, cmd)

                    t_last = rospy.get_time()

                # log data every loop
                self.logger.log(self.t_trial_start)

            elif self.state == "Resetting":
                # wait for some time or user input to start the next trial
                flag_start_trial = False
                if self.mode == "auto":
                    if rospy.get_time() - self.t_reset_start >= self.resetting_dt:
                        flag_start_trial = True
                else:
                    flag_start_trial = self.flag_start_trial

                if flag_start_trial:
                    self.start_trial()

            elif self.state == "Pausing":
                # wait for user input
                if self.flag_start_trial:
                    self.flag_start_trial = False

                    self.t_resume_start = rospy.get_time()
                    self.state = "Resuming"

            elif self.state == "Resuming":
                # wait for timer
                if rospy.get_time() - self.t_resume_start >= self.resuming_dt:
                    self.start_trial()

    def run(self, trial_start):
        key_thread = threading.Thread(target=self._monitor_key)
        loop_thread = threading.Thread(target=self._loop, args=[trial_start])
        key_thread.start()
        loop_thread.start()

        key_thread.join()
        loop_thread.join()


if __name__ == "__main__":
    rospy.init_node("random_guidance_exp")

    trial_start = rospy.get_param("~trial_start", 0)

    exp = RandomGuidanceExp()
    exp.run(trial_start)
