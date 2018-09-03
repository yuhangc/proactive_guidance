#!/usr/bin/env python

import numpy as np
import pickle


def wrap_to_pi(ang):
    return (ang + np.pi) % (2 * np.pi) - np.pi


def load_trial(root_path, trial_id, flag_transform=True, offsets=(2.13, 2.74, -2.3562, -1.5708)):
    data = np.loadtxt(root_path + "/trial" + str(trial_id) + ".txt", delimiter=", ")

    t = data[:, 0]
    pose = data[:, 1:4]
    pose[:, 2] = wrap_to_pi(np.deg2rad(pose[:, 2]))

    # optionally transform the coordinates
    if flag_transform:
        # positions
        offset = np.array([offsets[0], offsets[1]])
        th = offsets[2]

        rot_mat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pose[:, 0:2] = np.dot(pose[:, 0:2], rot_mat.transpose()) + offset

        # orientation
        pose[:, 2] = wrap_to_pi(-pose[:, 2] + offsets[3])

    return t, pose


def transform_to_body(pose):
    # compute the initial orientation by averaging the first 10 samples
    ths = pose[:10, 2]
    th_diff = ths - ths[0]

    th_diff = wrap_to_pi(th_diff)
    th_avg = ths[0] + np.mean(th_diff)
    pos_avg = np.mean(pose[:10, :2], axis=0)

    # compute pose in pose_init frame
    pose_trans = pose - np.array([pos_avg[0], pos_avg[1], th_avg])
    pose_trans[:, 2] = wrap_to_pi(pose_trans[:, 2])

    rot = np.array([[np.cos(th_avg), np.sin(th_avg)],
                    [-np.sin(th_avg), np.cos(th_avg)]])

    pose_trans[:, 0:2] = np.dot(pose_trans[:, 0:2], rot.transpose())

    return pose_trans


def load_save_all(root_path, protocol_file, n_trial, offsets=None, flag_transform_to_body=True):
    t_all = []
    pose_all =[]

    for trial in range(n_trial):
        if offsets is not None:
            t, pose = load_trial(root_path, trial, flag_transform=True, offsets=offsets)
        else:
            t, pose = load_trial(root_path, trial, flag_transform=False)

        # optionally transform to body frame
        if flag_transform_to_body:
            pose = transform_to_body(pose)

        t_all.append(t)
        pose_all.append(pose)

    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    # save to root path
    with open(root_path + "/raw_transformed.pkl", "w") as f:
        pickle.dump((t_all, pose_all, protocol_data), f)


if __name__ == "__main__":
    # load_save_all("/home/yuhang/Documents/proactive_guidance/training_data/test1-0830",
    #               "../../resources/protocols/random_continuous_protocol_10rep2.txt",
    #               120, (2.13, 2.74, -np.pi * 0.75, -np.pi * 0.5))

    load_save_all("/home/yuhang/Documents/proactive_guidance/training_data/user0/audio",
                  "../../resources/protocols/random_continuous_protocol_5rep2.txt",
                  120, (2.13, 3.0, -np.pi * 0.75, np.pi))
