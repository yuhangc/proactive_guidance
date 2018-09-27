#!/usr/bin/env python

import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


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


def load_save_all(root_path, protocol_file, flag_transform_to_body=True, rot_offset=-0.4):
    t_all = []
    pose_all =[]

    protocol_data = np.loadtxt(protocol_file, delimiter=", ")
    n_trial = len(protocol_data)

    for trial in range(n_trial):
        t, pose = load_trial(root_path, trial)
        pose[:, 2] += rot_offset

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

        traj = data[:, 1:]

        # extend the traj a little for visual effect
        s = traj[-1].copy()
        s_g = protocol_data[trial, 2:]

        d = np.linalg.norm(s[:2] - s_g)
        if d >= 0.3:
            th = np.arctan2(s_g[1] - s[1], s_g[0] - s[0])
            s += np.array([np.cos(th), np.sin(th), 0.0, 0.0, 0.0]) * 0.1
            traj = np.vstack((traj, s.reshape(1, -1)))

        t_all.append(data[:, 0])
        pose_all.append(traj)

    # visualize
    # generate a color map
    n_colors = int(np.max(protocol_data[:, 0]))
    cm = plt.get_cmap("gist_rainbow")

    fig, axes = plt.subplots()
    for trial in range(n_trial):
        traj = pose_all[trial]

        axes.plot(traj[:, 0], traj[:, 1], color=cm(1. * protocol_data[trial, 0] / n_colors))
        axes.axis("equal")

    # plot the goals
    visited = np.zeros((100, ))
    for trial in range(n_trial):
        trial_id = int(protocol_data[trial, 0])
        if visited[trial_id] < 1.0:
            visited[trial_id] = 1.0
            circ = Circle((protocol_data[trial, 2], protocol_data[trial, 3]), radius=0.35, facecolor='r', alpha=0.3)
            axes.add_patch(circ)

    plt.show()


def visualize_free_space_test_mixed(root_path, protocol_file):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_trial = len(protocol_data)

    pose_all_mpd = []
    pose_all_naive = []
    for trial in range(n_trial):
        data = np.loadtxt(root_path + "/trial" + str(trial) + ".txt", delimiter=", ")
        traj = data[:, 1:]

        # extend the traj a little for visual effect
        s = traj[-1].copy()
        s_g = protocol_data[trial, 2:]

        d = np.linalg.norm(s[:2] - s_g)
        if d >= 0.3:
            th = np.arctan2(s_g[1] - s[1], s_g[0] - s[0])
            s += np.array([np.cos(th), np.sin(th), 0.0, 0.0, 0.0]) * 0.1
            traj = np.vstack((traj, s.reshape(1, -1)))

        if protocol_data[trial, 1] == 0:
            pose_all_naive.append(traj)
        else:
            pose_all_mpd.append(traj)

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

    # plot the goals
    visited = np.zeros((100, ))
    for trial in range(n_trial):
        trial_id = int(protocol_data[trial, 0])
        if visited[trial_id] < 1.0:
            visited[trial_id] = 1.0
            circ = Circle((protocol_data[trial, 2], protocol_data[trial, 3]), radius=0.35, facecolor='r', alpha=0.3)
            axes[0].add_patch(circ)
            circ = Circle((protocol_data[trial, 2], protocol_data[trial, 3]), radius=0.35, facecolor='r', alpha=0.3)
            axes[1].add_patch(circ)

    axes[0].axis("equal")
    axes[1].axis("equal")

    axes[0].set_title("Naive Policy")
    axes[1].set_title("Optimized Policy")

    plt.show()


def load_free_space_test_mixed(root_path, protocol_file, n_cond=3, flag_all_mixed=True, mcts_proto=None):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")
    cond=["naive", "optimized", "as needed"]

    n_trial = len(protocol_data)

    n_target = int(max(protocol_data[:, 0])) + 1

    traj_all = []
    comm_all = []

    for i in range(n_cond):
        traj_cond = [[] for i in range(n_target)]
        comm_cond = [[] for i in range(n_target)]
        traj_all.append(traj_cond)
        comm_all.append(comm_cond)

    if flag_all_mixed:
        load_path = root_path + "/trials"
    else:
        load_path = root_path + "/naive_mdp"

    for trial in range(n_trial):
        data = np.loadtxt(load_path + "/trial" + str(trial) + ".txt", delimiter=", ")
        comm_data = np.loadtxt(load_path + "/trial" + str(trial) + "_comm.txt", delimiter=", ")

        cond = int(protocol_data[trial, 1])
        target_id = int(protocol_data[trial, 0])
        traj = data

        traj_all[cond][target_id].append(traj)
        comm_all[cond][target_id].append(comm_data)

    if not flag_all_mixed:
        # separately load the mcts case
        protocol_data = np.loadtxt(mcts_proto, delimiter=", ")
        n_trial = len(protocol_data)

        load_path = root_path + "/mcts"

        for trial in range(n_trial):
            data = np.loadtxt(load_path + "/trial" + str(trial) + ".txt", delimiter=", ")
            comm_data = np.loadtxt(load_path + "/trial" + str(trial) + "_comm.txt", delimiter=", ")

            target_id = int(protocol_data[trial, 0])
            traj = data

            traj_all[2][target_id].append(traj)
            comm_all[2][target_id].append(comm_data)

    with open(root_path + "/traj_raw.pkl", "w") as f:
        pickle.dump(traj_all, f)

    with open(root_path + "/comm_raw.pkl", "w") as f:
        pickle.dump(comm_all, f)


def load_planner_exp(root_path, protocol_file):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_trial = len(protocol_data)

    n_target = int(max(protocol_data[:, 0])) + 1

    traj_list = [[] for i in range(n_target)]
    comm_list = [[] for i in range(n_target)]
    for trial in range(n_trial):
        data = np.loadtxt(root_path + "/trial" + str(trial) + ".txt", delimiter=", ")
        comm_data = np.loadtxt(root_path + "/trial" + str(trial) + "_comm.txt", delimiter=", ")

        target_id = int(protocol_data[trial, 0])
        traj = data

        traj_list[target_id].append(traj)
        comm_list[target_id].append(comm_data)

    with open(root_path + "/traj_raw.pkl", "w") as f:
        pickle.dump(traj_list, f)

    with open(root_path + "/comm_raw.pkl", "w") as f:
        pickle.dump(comm_list, f)


def visualize_mixed_exp(root_path, protocol_file):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_trial = len(protocol_data)

    # generate a color map
    n_colors = int(np.max(protocol_data[:, 0]))
    cm = plt.get_cmap("gist_rainbow")

    n_cond = 3
    n_target = 7
    cond_name = ["naive", "optimized", "as needed"]
    # pose_all = [[] for i in range(n_cond)]

    fig, axes = plt.subplots(1, n_cond, figsize=(16, 5))

    # plot the goals
    target_pos = np.zeros((n_target, 2))
    visited = np.zeros((100, ))

    for trial in range(n_trial):
        trial_id = int(protocol_data[trial, 0])
        if visited[trial_id] < 1.0:
            visited[trial_id] = 1.0
            target_pos[trial_id] = protocol_data[trial, 2:4]
            for i in range(n_cond):
                circ = Circle((protocol_data[trial, 2], protocol_data[trial, 3]),
                              radius=0.35, facecolor='w', alpha=1.0, edgecolor='k')
                axes[i].add_patch(circ)

    with open(root_path + "/traj_raw.pkl") as f:
        traj_data = pickle.load(f)

    for cond in range(n_cond):
        for target in range(n_target):
            for traj in traj_data[cond][target]:
                # extend the traj a little for visual effect
                s = traj[-1].copy()
                s_g = target_pos[target]

                d = np.linalg.norm(s[:2] - s_g)
                if d >= 0.3:
                    th = np.arctan2(s_g[1] - s[1], s_g[0] - s[0])
                    s += np.array([np.cos(th), np.sin(th), 0.0, 0.0, 0.0, 0.0]) * 0.1
                    traj = np.vstack((traj, s.reshape(1, -1)))

                axes[cond].plot(traj[:, 1], traj[:, 2], color=cm(1. * target / n_colors))

    # for trial in range(n_trial):
    #     data = np.loadtxt(root_path + "/trials/trial" + str(trial) + ".txt", delimiter=", ")
    #     traj = data[:, 1:]
    #
    #     # extend the traj a little for visual effect
    #     s = traj[-1].copy()
    #     s_g = protocol_data[trial, 2:]
    #
    #     d = np.linalg.norm(s[:2] - s_g)
    #     if d >= 0.3:
    #         th = np.arctan2(s_g[1] - s[1], s_g[0] - s[0])
    #         s += np.array([np.cos(th), np.sin(th), 0.0, 0.0, 0.0]) * 0.1
    #         traj = np.vstack((traj, s.reshape(1, -1)))
    #
    #     pid = protocol_data[trial, 1]
    #     axes[pid].plot(traj[:, 0], traj[:, 1], color=cm(1. * protocol_data[trial, 0] / n_colors))

    for i in range(n_cond):
        axes[i].axis("equal")
        axes[i].set_title(cond_name[i])

    plt.show()


if __name__ == "__main__":
    # load_save_all("/home/yuhang/Documents/proactive_guidance/training_data/test1-0830",
    #               "../../resources/protocols/random_continuous_protocol_10rep2.txt",
    #               120, (2.13, 2.74, -np.pi * 0.75, -np.pi * 0.5))

    # load_save_all("/home/yuhang/Documents/proactive_guidance/training_data/user1/haptic",
    #               "../../resources/protocols/random_continuous_protocol_5rep2.txt",
    #               flag_transform_to_body=False)

    # load_free_space_test("/home/yuhang/Documents/proactive_guidance/planner_exp/user0",
    #                      "../../resources/protocols/free_space_exp_protocol_7targets_mdp.txt")

    # visualize_free_space_test_mixed("/home/yuhang/Documents/proactive_guidance/test_free_space/user0",
    #                                 "../../resources/protocols/free_space_exp_protocol_7targets_mixed.txt")

    # load_free_space_test_mixed("/home/yuhang/Documents/proactive_guidance/test_free_space/user3",
    #                            "../../resources/protocols/free_space_exp_protocol_7targets_mixed.txt")
    #
    # load_planner_exp("/home/yuhang/Documents/proactive_guidance/planner_exp/user3",
    #                  "../../resources/protocols/free_space_exp_protocol_7targets_mdp.txt")

    # load_random_guidance_exp("/home/yuhang/Documents/proactive_guidance/training_data/user0/random_guidance", 30)

    visualize_mixed_exp("/home/yuhang/Documents/proactive_guidance/planner_exp/user3",
                        "../../resources/protocols/free_space_exp_protocol_7targets_mixed.txt")

    # load_free_space_test_mixed("/home/yuhang/Documents/proactive_guidance/planner_exp/user7",
    #                            "../../resources/protocols/free_space_exp_protocol_7targets_mixed.txt")

    # load_free_space_test_mixed("/home/yuhang/Documents/proactive_guidance/planner_exp/user3",
    #                            "../../resources/protocols/free_space_exp_protocol_7targets_mixed2.txt",
    #                            flag_all_mixed=False,
    #                            mcts_proto="../../resources/protocols/free_space_exp_protocol_7targets_mdp.txt")
