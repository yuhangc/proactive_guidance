#!/usr/bin/env python
import numpy as np
import Queue

import rospy
from std_msgs.msg import Float32MultiArray
from people_msgs.msg import PositionMeasurementArray


class DataLogger(object):
    def __init__(self, save_path, flag_log_to_file, file_name="test.txt"):
        self.save_path = save_path
        self.flag_log_to_file = flag_log_to_file

        self.human_pose = np.zeros((3,))
        self.pose_latest = np.zeros((3, ))
        self.human_vel = np.zeros((3, ))
        self.cal_data = np.zeros((4, ))
        self.t_meas = rospy.get_time()

        self.human_pose_hist = []
        self.human_vel_hist = []
        self.cal_hist = []
        self.t_hist = []
        self.comm_hist = []

        # queues to temporarily store data
        self.pos_queue = Queue.Queue(maxsize=20)
        self.rot_queue = Queue.Queue(maxsize=20)

        # subscribers
        self.human_pos_sub = rospy.Subscriber("/people_tracker_measurements", PositionMeasurementArray,
                                              self.people_tracking_callback)
        self.human_rot_sub = rospy.Subscriber("/human_rotation", Float32MultiArray,
                                              self.people_rot_callback)

        self.valid_range_x = [0.2, 10.0]
        self.valid_range_y = [0.2, 10.0]
        self.robot_pose = np.zeros((3, ))

        self.human_rot_inversion = -1.0
        self.human_rot_offset = 188.0

        # open file if in direct mode
        if self.flag_log_to_file:
            self.save_file = open(self.save_path + "/" + file_name)

    def load_env_config(self, robot_pose, ranges):
        self.robot_pose = robot_pose
        self.valid_range_x, self.valid_range_y = ranges

    def log(self, t_start):
        while (not self.pos_queue.empty()) and (not self.rot_queue.empty()):
            self.t_meas, self.human_pose[0], self.human_pose[1] = self.pos_queue.get()
            self.human_pose[2] = self.rot_queue.get()

            if self.flag_log_to_file:
                self.save_file.write("{:f}, {:f}, {:f}, {:f}\n".format(self.t_meas-t_start, self.human_pose[0],
                                                                       self.human_pose[1], self.human_pose[2]))
            else:
                self.t_hist.append([self.t_meas - t_start])
                self.human_pose_hist.append(self.human_pose.copy())

    def log_comm(self, t_comm, comm):
        self.comm_hist.append([t_comm, comm])

    def save_data(self, file_name="", flag_save_comm=False):
        if self.flag_log_to_file:
            self.save_file.close()
        else:
            data = np.hstack((np.asarray(self.t_hist), np.asarray(self.human_pose_hist)))
            if flag_save_comm:
                np.savetxt(self.save_path + "/" + file_name + ".txt", data, fmt="%.3f", delimiter=", ")
                np.savetxt(self.save_path + "/" + file_name + "_comm.txt", np.asarray(self.comm_hist), fmt="%.3f", delimiter=", ")
            else:
                np.savetxt(self.save_path + "/" + file_name, data, fmt="%.3f", delimiter=", ")

    def reset(self):
        self.human_pose_hist = []
        self.human_vel_hist = []
        self.t_hist = []
        self.comm_hist = []
        with self.pos_queue.mutex:
            self.pos_queue.queue.clear()
        with self.rot_queue.mutex:
            self.rot_queue.queue.clear()

    def get_pose(self):
        return self.pose_latest

    def filter_measurement(self, position):
        # first convert from odom frame to "world" frame
        pos = np.array([position.x, position.y])

        th = self.robot_pose[2]
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pos = rot.dot(pos) # + self.robot_pose[:2]

        if pos[0] < self.valid_range_x[0] or pos[0] > self.valid_range_x[1] or \
                        pos[1] < self.valid_range_y[0] or pos[1] > self.valid_range_y[1]:
            return False, pos

        return True, pos

    def people_tracking_callback(self, tracking_msg):
        # filter out outliers
        for people in tracking_msg.people:
            flag_inlier, pos_trans = self.filter_measurement(people.pos)
            if flag_inlier:
                # self.t_meas = tracking_msg.header.stamp.to_sec()
                # self.human_pose[0] = people.pos.x
                # self.human_pose[1] = people.pos.y

                # push into queue
                if self.pos_queue.full():
                    self.pos_queue.get()
                self.pos_queue.put_nowait((people.header.stamp.to_sec(), pos_trans[0], pos_trans[1]))

                self.pose_latest[0] = pos_trans[0]
                self.pose_latest[1] = pos_trans[1]

                break

    def people_rot_callback(self, rot_msg):
        for i in range(4):
            self.cal_data[i] = rot_msg.data[i+1]

        # correct rotation
        rot = (rot_msg.data[0] - self.human_rot_offset) * self.human_rot_inversion

        if self.rot_queue.full():
            self.rot_queue.get()
        self.rot_queue.put_nowait(rot)

        self.pose_latest[2] = np.deg2rad(rot)


if __name__ == "__main__":
    rospy.init_node("exp_data_logger")

    save_path = rospy.get_param("~save_path", ".")
    logger = DataLogger(save_path, True, file_name="test.txt")

    rate = rospy.Rate(40)
    t_start = rospy.get_time()
    while not rospy.is_shutdown():
        logger.log(t_start)
        rate.sleep()

    logger.save_data()
