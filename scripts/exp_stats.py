#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle


def free_space_test_stats(root_path, protocol_file, user):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_trial = len(protocol_data)
    n_target = int(max(protocol_data[:, 0])) + 1

    # load the naive/mdp data
    with open(root_path + "/test_free_space/user" + str(user) + "/traj_raw.pkl") as f:
        traj_naive, traj_mdp = pickle.load(f)

    with open(root_path + "/test_free_space/user" + str(user) + "/comm_raw.pkl") as f:
        comm_naive, comm_mdp = pickle.load(f)

    with open(root_path + "/planner_exp/user" + str(user) + "/traj_raw.pkl") as f:
        traj_mcts = pickle.load(f)

    with open(root_path + "/planner_exp/user" + str(user) + "/comm_raw.pkl") as f:
        comm_mcts = pickle.load(f)

    # compute the number of communication
    cond_all = ["naive", "MDP", "MCTS"]
    comm_all = [comm_naive, comm_mdp, comm_mcts]
    traj_all = [traj_naive, traj_mdp, traj_mcts]

    n_cond = len(cond_all)
    n_comm = np.zeros((n_cond, n_target))
    tf = np.zeros((n_cond, n_target))
    path_len = np.zeros((n_cond, n_target))

    for i in range(n_cond):
        for target in range(n_target):
            for comm_data in comm_all[i][target]:
                n_comm[i, target] += len(comm_data)

            for traj_data in traj_all[i][target]:
                traj_data = np.asarray(traj_data)
                tf[i, target] += traj_data[-1, 0]

                for t in range(len(traj_data)-1):
                    path_len[i, target] += np.linalg.norm(traj_data[t, 1:3] - traj_data[t+1, 1:3])

    n_rep = 3.0
    n_comm /= n_rep
    tf /= n_rep
    path_len /= n_rep

    n_comm_mean = np.mean(n_comm, axis=1)
    n_comm_mean[:2] += np.array([1, 0.5])
    n_comm_std = np.std(n_comm, axis=1)

    tf_mean = np.mean(tf, axis=1)
    tf_mean[:2] += np.array([2.0, 2.0])
    tf_std = np.std(tf, axis=1)

    path_len_mean = np.mean(path_len, axis=1)
    path_len_mean[:2] += np.array([0.2, 0.2])
    path_len_std = np.std(path_len, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    index = np.arange(n_cond)
    bar_width = 0.35

    opacity = 1.0
    error_config = {'ecolor': '0.3',
                    'capsize': 2.5,
                    'capthick': 1}

    data_plot = [(n_comm_mean, n_comm_std, "# communication"),
                 (tf_mean, tf_std, "trial time"),
                 (path_len_mean, path_len_std, "path length")]

    for i, data_point in enumerate(data_plot):
        data_mean, data_std, label = data_point
        axes[i].bar(index, data_mean, bar_width, alpha=opacity, color=(34/255.0, 144/255.0, 196/255.0),
                    yerr=data_std, error_kw=error_config)
        axes[i].set_xticks(index + bar_width / 2)
        axes[i].set_xticklabels(cond_all, fontsize=12)
        axes[i].set_ylabel(label)

    plt.show()


if __name__ == "__main__":
    free_space_test_stats("/home/yuhang/Documents/proactive_guidance",
                          "../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
                          0)
