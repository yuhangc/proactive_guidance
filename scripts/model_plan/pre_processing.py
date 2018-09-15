#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle


def wrap_to_pi(ang):
    return (ang + np.pi) % (2 * np.pi) - np.pi


def pre_processing(root_path, usr, cond, flag_visualize=True):
    # first process the one-step case
    # load the preprocessed data
    path = root_path + "/" + usr + "/" + cond
    data_file = path + "/raw_transformed.pkl"

    with open(data_file) as f:
        t_all, pose_all, protocol_data = pickle.load(f)

    # compute final travel distance and and directions
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
    mag_out = np.asarray(mag_out)

    # wrap the dir_out around at end points (-pi and pi)
    for i in range(len(dir_out)):
        if dir_out[i] - dir_in[i] > np.pi:
            dir_out[i] -= 2.0 * np.pi
        elif dir_out[i] - dir_in[i] < -np.pi:
            dir_out[i] += 2.0 * np.pi

    # create 2 different plots
    if flag_visualize:
        # plot all traj
        fig, ax = plt.subplots()
        for pose in pose_all:
            ax.plot(pose[:, 0], pose[:, 1])
        ax.axis("equal")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].scatter(np.rad2deg(dir_in), np.rad2deg(dir_out))
        axes[1].scatter(np.rad2deg(dir_in), mag_out)

        plt.show()

    # save data
    with open(path + "/processed.pkl", 'w') as f:
        pickle.dump(np.hstack((dir_in.reshape(-1, 1), dir_out.reshape(-1, 1), mag_out.reshape(-1, 1))), f)


def visualize_processes(root_path, usr, cond):
    # load the preprocessed data
    path = root_path + "/user" + str(usr) + "/" + cond
    data_file = path + "/raw_transformed.pkl"

    with open(data_file) as f:
        t_all, pose_all, protocol_data = pickle.load(f)

    # group data by feedback
    input_mapping = {}
    for t, pose, protocol in zip(t_all, pose_all, protocol_data):
        alpha_d = protocol[0] - 90.0
        if alpha_d > 180:
            alpha_d -= 360

        if input_mapping.has_key(alpha_d):
            input_mapping[alpha_d].append((t, pose))
        else:
            input_mapping[alpha_d] = [(t, pose)]

    # create subplots for each input
    n_dir = 24
    n_col = 6
    n_row = 4

    fig1, axes1 = plt.subplots(n_row, n_col, figsize=(20, 12))
    fig2, axes2 = plt.subplots(n_row, n_col, figsize=(20, 12))

    i = 0
    j = 0
    for key in sorted(input_mapping.iterkeys()):
        for data in input_mapping[key]:
            t, pose = data
            t = t - t[0]

            pose[:, 2] = wrap_to_pi(pose[:, 2])
            axes1[i][j].plot(t, pose[:, 2])

            d = np.linalg.norm(pose[:, :2], axis=1)
            axes2[i][j].plot(t, d)

        j += 1
        if j >= n_col:
            i += 1
            j = 0

    plt.show()


def pre_processing_random_guidance(root_path, usr, flag_visualize=True):
    # first process the one-step case
    # load the preprocessed data
    path = root_path + "/user" + str(usr) + "/random_guidance"
    data_file = path + "/raw.pkl"

    with open(data_file, "w") as f:
        t_all, pose_all, comm_all = pickle.load(f)

    n_trial = len(pose_all)

    dir_in = []
    dir_out = []

    for trial in range(n_trial):
        # trajectories are in body frame
        # divide into segments based on communication
        t = t_all[trial]
        pose = pose_all[trial]
        comm = comm_all[trial]

        # fix timing issue?
        t -= t[0]


if __name__ == "__main__":
    # pre_processing("/home/yuhang/Documents/proactive_guidance/training_data", 0, "one_step")
    pre_processing("/home/yuhang/Documents/proactive_guidance/training_data", "pilot0", "audio")
    # pre_processing("/home/yuhang/Documents/proactive_guidance/training_data", 0, "audio")
    # visualize_processes("/home/yuhang/Documents/proactive_guidance/training_data", 0, "continuous")

