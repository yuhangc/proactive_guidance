#!/usr/bin/env python
import numpy as np

import rospy
from std_msgs.msg import Float32MultiArray
from people_msgs.msg import PositionMeasurementArray


class DataLogger(object):
    def __init__(self, save_path, flag_log_to_file, file_name="test.txt"):
        self.save_path = save_path
        self.flag_log_to_file = flag_log_to_file

        self.human_pose = np.zeros((3,))
        self.human_vel = np.zeros((3, ))
        self.t_meas = rospy.get_time()

        self.human_pose_hist = []
        self.human_vel_hist = []
        self.t_hist = []

        # subscribers
        self.human_pos_sub = rospy.Subscriber("/people_tracker_measurements", PositionMeasurementArray,
                                              self.people_tracking_callback)
        self.human_rot_sub = rospy.Subscriber("/human_rotation", Float32MultiArray,
                                              self.people_rot_callback)

        self.valid_range_x = (0.2, 10.0)
        self.valid_range_y = (0.2, 10.0)

        # open file if in direct mode
        if self.flag_log_to_file:
            self.save_file = open(self.save_path + "/" + file_name)

    def log(self, t_start):
        if self.flag_log_to_file:
            self.save_file.write("{:f}, {:f}, {:f}, {:f}\n".format(self.t_meas-t_start, self.human_pose[0],
                                                                   self.human_pose[1], self.human_pose[2]))
        else:
            self.t_hist.append([self.t_meas-t_start])
            self.human_pose_hist.append(self.human_pose.copy())

    def save_data(self, file_name=""):
        if self.flag_log_to_file:
            self.save_file.close()
        else:
            data = np.hstack((np.asarray(self.t_hist), np.asarray(self.human_pose_hist)))
            np.savetxt(self.save_path + "/" + file_name, data, fmt="%.3f", delimiter=", ")

    def reset(self):
        self.human_pose_hist = []
        self.human_vel_hist = []
        self.t_hist = []

    def filter_measurement(self, position):
        # first convert from odom frame to "world" frame
        pos = np.array([position.x, position.y])

        th = np.pi / 4.0
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pos = rot.dot(pos)

        if pos[0] < self.valid_range_x[0] or pos[0] > self.valid_range_x[1] or \
                        pos[1] < self.valid_range_y[0] or pos[1] > self.valid_range_y[1]:
            return False

        return True

    def people_tracking_callback(self, tracking_msg):
        # filter out outliers
        for people in tracking_msg.people:
            if self.filter_measurement(people.pos):
                self.t_meas = rospy.get_time()
                self.human_pose[0] = people.pos.x
                self.human_pose[1] = people.pos.y
                break

    def people_rot_callback(self, rot_msg):
        self.human_pose[2] = rot_msg.data[0]


if __name__ == "__main__":
    rospy.init_node("exp_data_logger")

    save_path = rospy.get_param("~save_path", ".")
    logger = DataLogger(save_path, True, file_name="test.txt")

    rate = rospy.Rate(40)
    while not rospy.is_shutdown():
        logger.log()
        rate.sleep()

    logger.save_data()
