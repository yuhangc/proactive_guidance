#!/usr/bin/env python

import numpy as np
import pickle

import matplotlib.pyplot as plt


def wrap_to_pi(ang):
    return (ang + np.pi) % (2 * np.pi) - np.pi


def load_trial(root_path, trial_id):
    data = np.loadtxt(root_path + "/trial" + str(trial_id) + ".txt", delimiter=", ")

    t = data[:, 0]
    pose = data[:, 1:4]
    pose[:, 2] = wrap_to_pi(np.deg2rad(pose[:, 2]))

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


def load_save_all(root_path, protocol_file, flag_transform_to_body=True):
    t_all = []
    pose_all =[]

    protocol_data = np.loadtxt(protocol_file, delimiter=", ")
    n_trial = len(protocol_data)

    for trial in range(n_trial):
        t, pose = load_trial(root_path, trial)

        # optionally transform to body frame
        if flag_transform_to_body:
            pose = transform_to_body(pose)

        t_all.append(t)
        pose_all.append(pose)

    # save to root path
    with open(root_path + "/raw_transformed.pkl", "w") as f:
        pickle.dump((t_all, pose_all, protocol_data), f)


def load_random_guidance_exp(root_path, n_trials):
    t_all = []
    pose_all = []
    comm_all = []

    for trial in range(n_trials):
        # load data
        data = np.loadtxt(root_path + "/trial" + str(trial) + ".txt", delimiter=", ")
        t_all.append(data[:, 0])

        pose = data[:, 1:]
        pose[:, 2] = wrap_to_pi(np.deg2rad(pose[:, 2]))
        pose_all.append(transform_to_body(pose))

        data = np.loadtxt(root_path + "/trial" + str(trial) + "_comm.txt", delimiter=", ")
        comm_all.append(data)

    # save to root path
    with open(root_path + "/raw_random_guidance.pkl", "w") as f:
        pickle.dump((t_all, pose_all, comm_all), f)

    # some visualization to make sure things make sense
    fig, ax = plt.subplots()
    for pose in pose_all:
        ax.plot(pose[:, 0], pose[:, 1])

    plt.show()


def load_free_space_test(root_path, protocol_file):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_trial = len(protocol_data)

    t_all = []
    pose_all =[]
    for trial in range(n_trial):
        data = np.loadtxt(root_path + "/trial" + str(trial) + ".txt", delimiter=", ")
        t_all.append(data[:, 0])
        pose_all.append(data[:, 1:])

    # visualize
    # generate a color map
    n_colors = int(np.max(protocol_data[:, 0]))
    cm = plt.get_cmap("gist_rainbow")

    fig, axes = plt.subplots()
    for trial in range(n_trial):
        traj = pose_all[trial]

        axes.plot(traj[:, 0], traj[:, 1], color=cm(1. * protocol_data[trial, 0] / n_colors))
        axes.axis("equal")

    plt.show()


def load_free_space_test_mixed(root_path, protocol_file):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_trial = len(protocol_data)

    pose_all_mpd = []
    pose_all_naive = []
    for trial in range(n_trial):
        data = np.loadtxt(root_path + "/trial" + str(trial) + ".txt", delimiter=", ")
        if protocol_data[trial, 1] == 0:
            pose_all_naive.append(data[:, 1:])
        else:
            pose_all_mpd.append(data[:, 1:])

    # visualize
    # generate a color map
    n_colors = int(np.max(protocol_data[:, 0]))
    cm = plt.get_cmap("gist_rainbow")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    i_naive = 0
    i_mdp = 0
    for trial in range(n_trial):
        if protocol_data[trial, 1] == 0:
            traj = pose_all_naive[i_naive]
            i_naive += 1
            axes[0].plot(traj[:, 0], traj[:, 1], color=cm(1. * protocol_data[trial, 0] / n_colors))
        else:
            traj = pose_all_mpd[i_mdp]
            i_mdp += 1
            axes[1].plot(traj[:, 0], traj[:, 1], color=cm(1. * protocol_data[trial, 0] / n_colors))

    axes[0].axis("equal")
    axes[1].axis("equal")

    plt.show()


if __name__ == "__main__":
    # load_save_all("/home/yuhang/Documents/proactive_guidance/training_data/test1-0830",
    #               "../../resources/protocols/random_continuous_protocol_10rep2.txt",
    #               120, (2.13, 2.74, -np.pi * 0.75, -np.pi * 0.5))

    load_save_all("/home/yuhang/Documents/proactive_guidance/training_data/user1/haptic",
                  "../../resources/protocols/random_continuous_protocol_5rep2.txt",
                  flag_transform_to_body=False)

    # load_free_space_test("/home/yuhang/Documents/proactive_guidance/test_free_space/user0-0912/mdp",
    #                      "../../resources/protocols/free_space_exp_protocol_7targets.txt")

    # load_free_space_test_mixed("/home/yuhang/Documents/proactive_guidance/test_free_space/user0-0912/mixed",
    #                            "../../resources/protocols/free_space_exp_protocol_7targets_mixed.txt")

    # load_random_guidance_exp("/home/yuhang/Documents/proactive_guidance/training_data/user0/random_guidance", 30)
