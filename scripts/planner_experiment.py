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
from model_plan.policies import MDPFixedTimePolicy
from model_plan.planner import Planner, PlannerNaive
from exp_training.data_recorder import DataLogger
from exp_training.utils import getKey


class PlannerExperiment(object):
    def __init__(self):
        # read in configurations
        # create planner
        planner = rospy.get_param("~planner", "mcts")

        if planner == "mcts":
            self.planner = Planner()
        else:
            self.planner = PlannerNaive()

        # directory that stores the policies for each trial
        self.policy_dir = rospy.get_param("~planner_dir", "mdp_haptic")

        # feedback modality
        self.modality = rospy.get_param("~modality", "haptic")
        self.default_policy = rospy.get_param("~policy", "mdp")
        self.env = rospy.get_param("~env", "free_space")

        # time interval to update plan
        self.t_plan_last = 0.0
        self.planner_dt = rospy.get_param("~planner_dt", 1.0)
        self.t_plan_max = rospy.get_param("~t_plan_max", 0.5)

        self.dx_plan_th = rospy.get_param("~dx_plan_th", 0.2)
        self.dx_alp_th = rospy.get_param("~dx_alp_th", 0.2)

        self.t_trial_max = rospy.get_param("~t_trial_max", 25.0)

        # optimal plan
        self.a_opt = None

        self.s_last_comm = np.array([5.0, 5.0, 0.0])
        self.s_last_alp = None

        # time interval to check stop and update planner measurement
        self.t_meas_last = 0.0
        self.meas_update_dt = rospy.get_param("~meas_update_dt", 0.1)

        # time interval for resetting/pausing
        self.starting_dt = rospy.get_param("~starting_dt", 2.0)
        self.resetting_dt = rospy.get_param("~resetting_dt", 10.0)
        self.resuming_dt = rospy.get_param("~resuming_dt", 5.0)
        self.t_reset_start = 0.0
        self.t_resume_start = 0.0
        self.t_trial_start = 0.0
        self.t_starting_start = 0.0
        
        self.alp_start = 0.0
        self.alp_start_count = 0

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

        self.flag_compute_plan = False
        self.flag_plan_generated = False
        self.flag_is_waiting = False

        # valid states are "Running", "Pausing", "Resetting", "Resuming"
        self.state = "Pausing"

        # create publisher
        self.haptic_msg_pub = rospy.Publisher("/haptic_control", String, queue_size=1)

    def publish_haptic_control(self, ctrl):
        # publish haptic feedback
        haptic_msg = String()
        haptic_msg.data = "{:d},{:d}".format(int(ctrl[0]), int(ctrl[1]))

        self.haptic_msg_pub.publish(haptic_msg)

        cmd = ctrl[0] - 90.0
        if cmd > 270:
            cmd -= 360

        print "{:d},{:d}\r".format(int(cmd), int(ctrl[1]))

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
        # print "human pose is: ", s, "\r"
        # print self.proto_data[self.trial, 2:4], "\r"
        # print "err is: ", err, "\r"

        if err < self.goal_reaching_th:
            # send a stop command
            print "Goal reached!\r"
            self.publish_haptic_control(self.msg_stop)

            return True
        else:
            # check for time limit
            if rospy.get_time() - self.t_trial_start > self.t_trial_max:
                print "Time limit exceeded...\r"
                self.publish_haptic_control(self.msg_stop)
                
                return True

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

        policy_dir = self.policy_dir + "/" + self.default_policy + "_" + self.modality + "/" + self.env

        with open(policy_dir + "/target" + str(target_id) + ".pkl") as f:
            default_policy = pickle.load(f)

        # load default policy into planner
        self.planner.create_policy(default_policy, self.modality)
        
        print "preparing to start trial...\r"

        self.alp_start = 0.0
        self.alp_start_cout = 0
        
        self.state = "Starting"
        self.t_starting_start = rospy.get_time()

    def _planner_thread(self):
        while not rospy.is_shutdown() and not self.flag_end_program:
            if self.flag_compute_plan:
                self.flag_compute_plan = False

                self.a_opt = self.planner.compute_plan(t_max=self.t_plan_max,
                                                       flag_with_prediction=True,
                                                       flag_wait_for_t_max=True)
                self.flag_plan_generated = True
            else:
                # simply do nothing?
                pass

    def _exp_thread(self, trial_start):
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
        print "Trial will start in ", self.resetting_dt, " seconds...\r"

        self.state = "Resetting"
        self.flag_start_trial = False
        self.t_reset_start = rospy.get_time()

        while not rospy.is_shutdown() and not self.flag_end_program:
            if self.state == "Starting":
                # keep monitoring the orientation and average up
                pose = self.logger.get_pose()

                self.alp_start += pose[2]
                self.alp_start_count += 1

                # check timer
                if rospy.get_time() - self.t_starting_start:
                    self.alp_start /= float(self.alp_start_count)
                    self.logger.adjust_rot_offset(self.alp_start)
                    
                    pose[2] -= self.alp_start

                    print "adjusting orientation offset by: ", self.alp_start, "\r"
           
                    # reset data logger
                    self.logger.reset()

                    print "Starting trial ", self.trial, "...\r"
                    self.t_trial_start = rospy.get_time()

                    self.flag_compute_plan = False
                    self.flag_plan_generated = False
                    self.flag_is_waiting = False

                    self.s_last_alp = None

                    self.state = "Running"

            elif self.state == "Running":
                # log data every loop
                self.logger.log(self.t_trial_start)

                # get latest tracking
                pose = self.logger.get_pose()

                flag_stop = False
                # first check for stop
                if self.check_for_stop(pose):
                    print "Trial ", self.trial, " ended\r"

                    # save data first
                    self.logger.save_data(file_name="trial{}".format(self.trial), flag_save_comm=True, flag_save_extra=True)

                    # check if exp ends
                    self.trial += 1
                    if self.trial >= len(self.proto_data):
                        print "All trials finished!\r"
                        break

                    # prepare to reset
                    self.t_reset_start = rospy.get_time()
                    self.state = "Resetting"
                    flag_stop = True

                # update planner measurement
                t_curr = rospy.get_time()
                if t_curr - self.t_meas_last > self.meas_update_dt:
                    self.planner.update_state(pose, t_curr)
                    self.t_meas_last = t_curr

                    self.logger.log_extra(t_curr-self.t_trial_start, [self.planner.alp_d_mean, self.planner.alp_d_cov, self.planner.v])

                    # print "pose is: ", pose, "\r

                # update alpha estimate
                if self.s_last_alp is not None:
                    dx_alp = np.linalg.norm(pose[:2] - self.s_last_alp[:2])
                    if dx_alp >= self.dx_alp_th:
                        self.planner.update_alp(pose)
                        self.s_last_alp = pose.copy()
                else:
                    self.s_last_alp = pose.copy()

                # print "flag_stop is: ", flag_stop, "flag_is_waiting is: ", self.flag_is_waiting, "\r"
                if not flag_stop and not self.flag_is_waiting:
                    dx = np.linalg.norm(pose[:2] - self.s_last_comm[:2])
                    # print "dx is: ", dx, "\r"

                    # minimum 1 second interval and position has changed
                    if t_curr - self.t_plan_last >= self.planner_dt - self.t_plan_max and dx > self.dx_plan_th:
                        print "prepare to compute plan...\r"
                        # first update alpha
                        self.planner.update_alp(pose)
                        self.s_last_alp = pose.copy()

                        # tell planner thread to start plan
                        self.flag_compute_plan = True

                        # wait for plan
                        self.flag_is_waiting = True

                if self.flag_is_waiting and self.flag_plan_generated:
                    print "got plan\r"
                    self.flag_plan_generated = False
                    self.flag_is_waiting = False

                    if self.a_opt is not None:
                        self.planner.execute_plan(pose, self.a_opt)
                        print "action is: ", self.a_opt, "state is: ", pose, "\r"

                        # convert to right format and publish
                        self.publish_haptic_control([self.convert_feedback(self.a_opt), 2])

                        # log the feedback
                        self.logger.log_comm(rospy.get_time() - self.t_trial_start, self.a_opt)

                        self.t_plan_last = rospy.get_time()
                        self.s_last_comm = pose.copy()

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
            
            rate.sleep()

    def run(self, trial_start):
        key_thread = threading.Thread(target=self._monitor_key)
        exp_thread = threading.Thread(target=self._exp_thread, args=[trial_start])
        planner_thread = threading.Thread(target=self._planner_thread)

        key_thread.start()
        exp_thread.start()
        planner_thread.start()

        key_thread.join()
        exp_thread.join()
        planner_thread.join()


if __name__ == "__main__":
    rospy.init_node("planner_experiment")

    trial_start = rospy.get_param("~trial_start", 0)

    exp = PlannerExperiment()
    exp.run(trial_start)
