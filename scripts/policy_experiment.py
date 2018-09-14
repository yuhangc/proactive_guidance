#!/usr/bin/env python

import sys
sys.path.append("/home/yuhang/ros_dev/src/proactive_guidance/scripts/model_plan")

import numpy as np
import pickle
import threading

import rospy
from std_msgs.msg import String
from people_msgs.msg import PositionMeasurementArray

from model_plan.gp_model_approx import GPModelApproxBase
from model_plan.movement_model import MovementModel
from model_plan.policies import NaivePolicy, MDPFixedTimePolicy
from exp_training.data_recorder import DataLogger
from exp_training.utils import getKey


class PolicyExperiment(object):
    def __init__(self):
        # read in configurations
        # load planner
        self.planner = None
        self.planner_dir = rospy.get_param("~planner_dir", "naive_planners")
        # with open(planner_file) as f:
        #     self.planner = pickle.load(f)

        # feedback modality
        self.modality = rospy.get_param("~modality", "haptic")
        self.policy = rospy.get_param("~policy", "mixed")

        # time interval to update plan
        self.planner_dt = rospy.get_param("~planner_dt", 2.0)

        # time interval for resetting/pausing
        self.resetting_dt = rospy.get_param("~resetting_dt", 10.0)
        self.resuming_dt = rospy.get_param("~resuming_dt", 5.0)
        self.t_reset_start = 0.0
        self.t_resume_start = 0.0
        self.t_trial_start = 0.0

        # goal reaching threshold
        self.goal_reaching_th = rospy.get_param("~goal_reaching_th", 0.3)

        # create a data logger
        path_saving = rospy.get_param("~path_saving", ".")
        robot_pose = rospy.get_param("~robot_pose", [0.0, 1.0, np.pi * 0.5])
        x_range = rospy.get_param("~x_range", [-5.0, 5.0])
        y_range = rospy.get_param("~y_range", [-1.0, 6.0])

        self.logger = DataLogger(path_saving, False)
        self.logger.load_env_config(robot_pose, [x_range, y_range])

        # experiment mode
        self.mode = rospy.get_param("~mode", "manual")      # can be 'manual' or 'auto'

        # load protocol
        protocol_file = rospy.get_param("~protocol_file", "protocol.txt")
        self.proto_data = np.loadtxt(protocol_file, delimiter=", ")

        self.trial = 0

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
        err = np.linalg.norm(s[:2] - self.proto_data[self.trial, 2:4])
        print "human pose is: ", s, "\r"
        print self.proto_data[self.trial, 2:4], "\r"
        print "err is: ", err, "\r"

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

    def start_trial(self):
        self.flag_start_trial = False

        # load planner
        target_id = int(self.proto_data[self.trial, 0])

        if self.policy == "mixed":
            policy_id = int(self.proto_data[self.trial, 1])
            if policy_id == 0:
                planner_dir = self.planner_dir + "/naive_" + self.modality + "/free_space"
            else:
                planner_dir = self.planner_dir + "/mdp_" + self.modality + "/free_space"
        else:
            planner_dir = self.planner_dir + "/" + self.policy + "_" + self.modality + "/free_space"

        with open(planner_dir + "/target" + str(target_id) + ".pkl") as f:
            self.planner = pickle.load(f)

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
            if self.state == "Running":# get latest tracking
                # log data every loop
                self.logger.log(self.t_trial_start)
                
                pose = self.logger.get_pose()

                # first check for stop
                if self.check_for_stop(pose):
                    print "Trial ", self.trial, " ended\r"

                    # save data first
                    self.logger.save_data(file_name="trial{}".format(self.trial), flag_save_comm=True)

                    # check if exp ends
                    self.trial += 1
                    if self.trial >= len(self.proto_data):
                        print "All trials finished!\r"
                        break

                    # prepare to reset
                    self.t_reset_start = rospy.get_time()
                    self.state = "Resetting"
                elif rospy.get_time() - t_last >= self.planner_dt:
                    # compute and send policy
                    cmd = self.planner.sample_policy(pose)

                    # convert to right format and publish
                    self.publish_haptic_control([self.convert_feedback(cmd), 2])

                    # log the feedback
                    self.logger.log_comm(rospy.get_time() - self.t_trial_start, cmd)

                    t_last = rospy.get_time()

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
    rospy.init_node("policy_experiment")

    trial_start = rospy.get_param("~trial_start", 0)

    exp = PolicyExperiment()
    exp.run(trial_start)
