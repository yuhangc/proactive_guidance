#!/usr/bin/env python

import numpy as np
import pickle


def wrap_to_pi(ang):
    return (ang + np.pi) % (2 * np.pi ) - np.pi


def load_trial(root_path, trial_id, flag_transform=True, offsets=(3.0, 3.0, -2.3562)):
    data = np.loadtxt(root_path + "/trial" + str(trial_id) + ".txt", delimiter=", ")

    t = data[:, 0]
    pose = data[:, 1:4]

    # optionally transform the coordinates
    if flag_transform:
        offset = np.array([offsets[0], offsets[1]])
        th = offsets[2]

        rot_mat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pose[:, 0:2] = np.dot(pose[:, 0:2], rot_mat.transpose()) + offset

    return t, pose


def load_save_all(root_path, protocol_file, n_trial, offsets=None):
    t_all = []
    pose_all =[]

    for trial in range(n_trial):
        if offsets is not None:
            t, pose = load_trial(root_path, trial, flag_transform=True, offsets=offsets)
        else:
            t, pose = load_trial(root_path, trial, flag_transform=False)

        t_all.append(t)
        pose_all.append(pose)

    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    # save to root path
    with open(root_path + "/raw_transformed.pkl", "w") as f:
        pickle.dump((t_all, pose_all, protocol_data), f)


if __name__ == "__main__":
    load_save_all("/home/yuhang/Documents/proactive_guidance/training_data/test0-0820",
                  "../../resources/protocols/random_protocol_3rep.txt",
                  72, (3.0, 3.0, -np.pi * 0.75))
