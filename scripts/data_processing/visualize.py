#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def load_trial(root_path, trial_id, flag_transform=True):
    data = np.loadtxt(root_path + "/trial" + str(trial_id) + ".txt", delimiter=", ")

    t = data[:, 0]
    pose = data[:, 1:4]

    # optionally transform the coordinates
    if flag_transform:
        offset = np.array([3.0, 3.0])
        th = -np.pi * 0.75

        rot_mat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pose[:, 0:2] = np.dot(pose[:, 0:2], rot_mat.transpose()) + offset

    return t, pose


def plot_all(root_path):
    # load protocol
    protocol_file = "../../resources/protocols/random_protocol.txt"
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    dirs = protocol_data[:, 0]
    mags = protocol_data[:, 1]

    # load all and group by magnitudes and directions
    n_trials = len(dirs)

    t_all = []
    pose_all = []

    for mag in range(3):
        t_mag = []
        pose_mag = []
        for dir in range(8):
            t_mag.append([])
            pose_mag.append([])

        t_all.append(t_mag)
        pose_all.append(pose_mag)

    for trial in range(n_trials):
        t, pose = load_trial(root_path, trial)
        t_all[int(mags[trial])][int(dirs[trial])].append(t)
        pose_all[int(mags[trial])][int(dirs[trial])].append(pose)

    # create plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # generate a color map
    n_colors = 8
    cm = plt.get_cmap("gist_rainbow")

    for mag in range(3):
        for dir in range(8):
            for poses in pose_all[mag][dir]:
                axes[mag].plot(poses[:, 0], poses[:, 1], color=cm(1. * dir / n_colors))
        axes[mag].axis("equal")
        axes[mag].set_xlim(-1.5, 3.5)
        axes[mag].set_ylim(-2.5, 3.0)

    plt.show()


if __name__ == "__main__":
    plot_all("/home/yuhang/Documents/proactive_guidance/training_data/test0")
