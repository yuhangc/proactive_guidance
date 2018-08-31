#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

from loading import load_trial, wrap_to_pi


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
        for dir in range(0, 8):
            for poses in pose_all[mag][dir]:
                axes[mag].plot(poses[:, 0], poses[:, 1], color=cm(1. * dir / n_colors))
        axes[mag].axis("equal")
        axes[mag].set_xlim(-2.5, 2.5)
        axes[mag].set_ylim(-2.5, 2.5)

    plt.show()


def visualize_data_continuous(root_path):
    # load the preprocessed data
    data_file = root_path + "/raw_transformed.pkl"

    with open(data_file) as f:
        t_all, pose_all, protocol_data = pickle.load(f)

    # create 4 different plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # first simply plot out all trajectories
    for pose in pose_all:
        axes[0, 0].plot(pose[:, 0], pose[:, 1])

    # then plot dir vs. feedback
    dir_out = []
    mag_out = []
    th_out = []
    for pose in pose_all:
        # average the last 10 samples
        pos_avg = np.mean(pose[-10:, 0:2], axis=0)
        dir_out.append(np.arctan2(pos_avg[1], pos_avg[0]))
        mag_out.append(np.linalg.norm(pos_avg))
        th_out.append(wrap_to_pi(np.mean(pose[-10:, 2])))

    dir_in = protocol_data[:, 0] * np.pi / 180.0 - np.pi * 0.5

    dir_out = wrap_to_pi(np.asarray(dir_out))
    dir_in = wrap_to_pi(dir_in)
    axes[0, 1].scatter(np.rad2deg(dir_in), np.rad2deg(dir_out))

    # plot mag vs. feedback
    axes[1, 0].scatter(np.rad2deg(dir_in), np.asarray(mag_out))

    # plot th vs. feedback
    axes[1, 1].scatter(dir_out, np.asarray(th_out))

    # 3D scatter of dir vs. feedback
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ri = 5.0
    ro = 2.0
    x = (ri + ro * np.cos(dir_out)) * np.cos(dir_in)
    y = (ri + ro * np.cos(dir_out)) * np.sin(dir_in)
    z = ro * np.sin(dir_out)
    ax.scatter(x, y, z)

    plt.show()


if __name__ == "__main__":
    # plot_all("/home/yuhang/Documents/proactive_guidance/training_data/test0-0826")

    visualize_data_continuous("/home/yuhang/Documents/proactive_guidance/training_data/test1-0830")
