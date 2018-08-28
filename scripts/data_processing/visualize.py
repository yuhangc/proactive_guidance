#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from loading import load_trial


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


if __name__ == "__main__":
    plot_all("/home/yuhang/Documents/proactive_guidance/training_data/test0-0826")
