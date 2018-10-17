#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.signal import savgol_filter
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng

from model_plan.plotting_utils import *
from model_plan.gp_model_approx import GPModelApproxBase


def mixed_exp_stats(root_path, protocol_file, user, flag_visualization=True):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_trial = len(protocol_data)
    n_target = int(max(protocol_data[:, 0])) + 1

    # load the naive/mdp data
    with open(root_path + "/user" + str(user) + "/traj_raw.pkl") as f:
        traj_all = pickle.load(f)

    with open(root_path + "/user" + str(user) + "/comm_raw.pkl") as f:
        comm_all = pickle.load(f)

    # compute the number of communication
    cond_all = ["Naive", "Optimized", "As Needed"]
    # comm_all = [comm_naive, comm_mdp, comm_mcts]
    # traj_all = [traj_naive, traj_mdp, traj_mcts]

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
                tf[i, target] += traj_data[-1, 0] - traj_data[0, 0]

                for t in range(len(traj_data)-1):
                    path_len[i, target] += np.linalg.norm(traj_data[t, 1:3] - traj_data[t+1, 1:3])

    n_rep = 3.0
    n_comm /= n_rep
    tf /= n_rep
    path_len /= n_rep

    n_comm_mean = np.mean(n_comm, axis=1)
    # n_comm_mean[:2] += np.array([1, 0.5])
    n_comm_std = np.std(n_comm, axis=1)

    tf_mean = np.mean(tf, axis=1)
    # tf_mean[:2] += np.array([2.0, 2.0])
    tf_std = np.std(tf, axis=1)

    path_len_mean = np.mean(path_len, axis=1)
    # path_len_mean[:2] += np.array([0.2, 0.2])
    path_len_std = np.std(path_len, axis=1)

    if flag_visualization:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        index = np.arange(n_cond)
        bar_width = 0.35

        opacity = 1.0
        error_config = {'ecolor': '0.3',
                        'capsize': 2.5,
                        'capthick': 1}

        data_plot = [(n_comm_mean, n_comm_std, "# of Communication"),
                     (tf_mean, tf_std, "Trial Time (s)"),
                     (path_len_mean, path_len_std, "Path Length (m)")]

        for i, data_point in enumerate(data_plot):
            data_mean, data_std, label = data_point
            axes[i].bar(index, data_mean, bar_width, alpha=opacity, color=(34/255.0, 144/255.0, 196/255.0),
                        yerr=data_std, error_kw=error_config)
            axes[i].set_xticks(index + bar_width / 2)
            axes[i].set_xticklabels(cond_all, fontsize=16)
            axes[i].set_ylabel(label, fontsize=16)
            axes[i].set_xlim(-0.25, 2.5)

            turn_off_box(axes[i])
            set_tick_size(axes[i], 14)

        fig.tight_layout()
        plt.show()

    return n_comm_mean, tf_mean, path_len_mean


def obs_exp_stats(root_path, protocol_file, map_file, user):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_trial = len(protocol_data)
    n_target = int(max(protocol_data[:, 0])) + 1

    # load the naive/mdp data
    with open(root_path + "/user" + str(user) + "/traj_raw.pkl") as f:
        traj_all = pickle.load(f)

    with open(root_path + "/user" + str(user) + "/comm_raw.pkl") as f:
        comm_all = pickle.load(f)

    with open(map_file) as f:
        target_all, obs_all = pickle.load(f)

    # compute the number of communication
    cond_all = ["Naive", "Optimized", "As Needed"]

    n_cond = len(cond_all)
    n_comm = np.zeros((n_cond, n_target))
    path_len = np.zeros((n_cond, n_target))
    ccount = np.zeros((n_cond, n_target))

    for i in range(n_cond):
        for target in range(n_target):
            for comm_data in comm_all[i][target]:
                n_comm[i, target] += len(comm_data)

            for traj_data in traj_all[i][target]:
                traj_data = np.asarray(traj_data)

                for t in range(len(traj_data)-1):
                    path_len[i, target] += np.linalg.norm(traj_data[t, 1:3] - traj_data[t+1, 1:3])

                    for x, y, w, h in obs_all[target]:
                        xrel = traj_data[t, 1] - x
                        yrel = traj_data[t, 2] - y

                        if 0 <= xrel <= w and 0 <= yrel <= h:
                            ccount[i][target] += 1.0

    n_rep = 3.0
    n_comm /= n_rep
    ccount /= n_rep
    path_len /= n_rep

    n_comm_mean = np.mean(n_comm, axis=1)
    n_comm_mean[:2] += np.array([1, 0.5])
    n_comm_std = np.std(n_comm, axis=1)

    ccount_mean = np.mean(ccount, axis=1)
    ccount_std = np.std(ccount, axis=1)

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

    data_plot = [(n_comm_mean, n_comm_std, "# of Communication"),
                 (ccount_mean, ccount_std, "Collisions"),
                 (path_len_mean, path_len_std, "Path Length (m)")]

    for i, data_point in enumerate(data_plot):
        data_mean, data_std, label = data_point
        axes[i].bar(index, data_mean, bar_width, alpha=opacity, color=(34/255.0, 144/255.0, 196/255.0),
                    yerr=data_std, error_kw=error_config)
        axes[i].set_xticks(index + bar_width / 2)
        axes[i].set_xticklabels(cond_all, fontsize=16)
        axes[i].set_ylabel(label, fontsize=16)
        axes[i].set_xlim(-0.25, 2.5)

        turn_off_box(axes[i])
        set_tick_size(axes[i], 14)

    fig.tight_layout()
    plt.show()


def compute_model_stats(root_path, users, flag_with_box=True):
    # load all models
    modalities = ["haptic", "audio"]

    n_user = len(users)
    n_mod = len(modalities)

    linearity = np.zeros((n_mod, n_user))
    symmetry = np.zeros((n_mod, n_user))
    mi = np.zeros((n_mod, n_user))
    delay = np.zeros((n_mod, n_user))

    for iu, user in enumerate(users):
        for im, modality in enumerate(modalities):
            with open(root_path + "/user" + str(user) + "/gp_model_" + modality + ".pkl") as f:
                model = pickle.load(f)

            # compute linearity
            linearity[im, iu] = compute_linearity(model)
            symmetry[im, iu] = compute_symmetry(model)
            mi[im, iu] = compute_mutual_info(model)
            delay[im, iu] = compute_delay(root_path, user, modality)

    # visualize the results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # for im in range(n_mod):
    #     x = np.ones_like(linearity[im]) * (im+1)
    #     axes[0].scatter(x, delay[im], alpha=0.5, lw=0)
    #     axes[1].scatter(x, linearity[im], alpha=0.5, lw=0)
    #     axes[2].scatter(x, mi[im], alpha=0.5, lw=0)

    h = []
    medianprops = dict(linestyle='-', linewidth=2.0, color='k', alpha=0.5)
    whisprops = dict(linestyle='-', linewidth=1.5, color=(0.5, 0.5, 0.5))
    capprops = whisprops
    flierprops = dict(marker='o', markerfacecolor='k', markersize=5,
                      linestyle='none', alpha=0.5)
    widths = [0.5, 0.5]

    if flag_with_box:
        h.append(axes[0].boxplot(delay.transpose(), notch=False, vert=True, patch_artist=True, showfliers=True,
                                 widths=widths, medianprops=medianprops, whiskerprops=whisprops,
                                 capprops=capprops, flierprops=flierprops))
        h.append(axes[1].boxplot(linearity.transpose(), notch=False, vert=True, patch_artist=True, showfliers=True,
                                 widths=widths, medianprops=medianprops, whiskerprops=whisprops,
                                 capprops=capprops, flierprops=flierprops))
        h.append(axes[2].boxplot(mi.transpose(), notch=False, vert=True, patch_artist=True, showfliers=True,
                                 widths=widths, medianprops=medianprops, whiskerprops=whisprops,
                                 capprops=capprops, flierprops=flierprops))

    colors = [(.278, .635, .847), (1, .706, .29)]
    for hi in h:
        for i, patch in enumerate(hi["boxes"]):
            patch.set_linewidth(2.0)
            patch.set_edgecolor((0.4, 0.4, 0.4))
            patch.set_facecolor(colors[i])

    ylabels = ["Delay (s)", "Nonlinearity", "Mutual Information (bits)"]
    for i, ax in enumerate(axes):
        ax.set_ylabel(ylabels[i], fontsize=16)
        ax.set_xticklabels(["Haptic", "Verbal"], fontsize=16)

        set_tick_size(ax, 14)
        turn_off_box(ax)

    axes[1].set_yticks([0.0, 0.4, 0.8, 1.2])
    axes[2].set_yticks([2.6, 3.0, 3.4, 3.8, 4.2])

    fig.tight_layout()

    plt.show()


def compute_delay(root_path, user, modality):
    # load the preprocessed data
    path = root_path + "/user" + str(user) + "/" + modality
    data_file = path + "/raw_transformed.pkl"

    with open(data_file) as f:
        t_all, pose_all, protocol_data = pickle.load(f)

    t_delay_all = []
    for i in range(len(protocol_data)):
        alpha_d = protocol_data[i, 0] - 90.0
        if alpha_d > 180:
            alpha_d -= 360

        if np.abs(alpha_d) > 45:
            # compute angular velocity?
            om = np.diff(pose_all[i][:, 2]) / np.diff(t_all[i])
            om_smooth = savgol_filter(om, 41, 3)

            # plt.plot(om_smooth)
            # plt.plot(om)
            # plt.show()

            idx = np.where(np.abs(om_smooth) > 0.5)[0]

            if modality == "haptic":
                t_th = 0.6
            else:
                t_th = 1.0

            if len(idx) > 0 and t_all[i][idx[0]] - t_all[i][0] > t_th:
                t_delay_all.append(t_all[i][idx[0]] - t_all[i][0] - 0.1)

    return np.mean(np.asarray(t_delay_all))


def compute_linearity(model):
    x = np.linspace(-np.pi, np.pi, 100)

    linearity = 0.0

    for i in range(len(x)-1):
        xi = 0.5 * (x[i] + x[i+1])
        y, y_std = model.predict_fast(xi)[0]
        y -= xi

        linearity += y**2 * (x[i+1] - x[i])

    return linearity


def compute_symmetry(model):
    x = np.linspace(0, np.pi, 100)

    symmetry = 0.0

    for i in range(len(x) - 1):
        xi = 0.5 * (x[i] + x[i+1])
        y, y_std = model.predict_fast(xi)[0]
        y_n, y_n_std = model.predict_fast(-xi)[0]

        symmetry += (y + y_n)**2 * (x[i+1] - x[i])

    return symmetry


def compute_mutual_info(model):
    x = np.linspace(-np.pi, np.pi, 200)
    px = 0.5 / np.pi

    # first compute the entropy of y
    y = np.linspace(-np.pi, np.pi, 200)
    ent_y = 0.0

    for i in range(len(y) - 1):
        py = 0.0
        yi = 0.5 * (y[i] + y[i+1])
        for j in range(len(x) - 1):
            xi = 0.5 * (x[j] + x[j+1])
            y_mean, y_std = model.predict_fast(xi)[0]

            pyx = np.exp(-(yi - y_mean)**2 / 2.0 / y_std**2) / np.sqrt(2.0 * np.pi) / y_std
            py += pyx * px * (x[j+1] - x[j])

        ent_y += -py * np.log2(py) * (y[i+1] - y[i])

    # compute entropy of y|x
    ent_yx = 0.0
    for i in range(len(x) - 1):
        xi = 0.5 * (x[i] + x[i+1])
        y_mean, y_std = model.predict_fast(xi)[0]

        if y_std > 0.1:
            y_std -= 0.04

        hyx = 0.5 * np.log2(2.0 * np.pi * np.exp(1) * y_std**2)
        ent_yx += hyx * px * (x[i+1] - x[i])

    return ent_y - ent_yx


def mixed_exp_stats_all(root_path, protocol_file, users):
    n_comm = []
    tf = []
    path_len = []

    for user in users:
        n_comm_u, tf_u, path_len_u = mixed_exp_stats(root_path, protocol_file, user, flag_visualization=False)
        n_comm.append(n_comm_u)
        tf.append(tf_u)
        path_len.append(path_len_u)

    n_cond = 3
    cond_all = ["Naive\nPolicy", "Optimized\nPolicy", "Communicate\nAs Needed"]
    # colors = [(.3, .3, .3), (.306, .404, .631), (.941, .40, .40)]
    colors = [(.8, .8, .8), (.278, .635, .847), (1, .706, .29)]

    n_comm_mean = np.mean(n_comm, axis=0)
    n_comm_std = np.std(n_comm, axis=0)

    tf_mean = np.mean(tf, axis=0)
    tf_std = np.std(tf, axis=0)

    path_len_mean = np.mean(path_len, axis=0)
    path_len_std = np.std(path_len, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    index = np.arange(n_cond)
    bar_width = 0.6

    opacity = 1.0
    error_config = {'ecolor': '0.3',
                    'linewidth': 1.5,
                    'capsize': 2.5,
                    'capthick': 1.5}

    data_plot = [(n_comm_mean, n_comm_std, "# of Communication"),
                 (tf_mean, tf_std, "Trial Time (s)"),
                 (path_len_mean, path_len_std, "Path Length (m)")]

    for i, data_point in enumerate(data_plot):
        data_mean, data_std, label = data_point
        # axes[i].bar(index, data_mean, bar_width, alpha=opacity, color=(34/255.0, 144/255.0, 196/255.0),
        #             yerr=data_std, error_kw=error_config)
        for j in range(n_cond):
            axes[i].bar(index[j], data_mean[j], bar_width, alpha=opacity, color=colors[j], lw=2,
                        edgecolor=(0.4, 0.4, 0.4), yerr=data_std[j], error_kw=error_config)
        axes[i].set_xticks(index + bar_width / 2)
        axes[i].set_xticklabels(cond_all, fontsize=16)
        axes[i].set_ylabel(label, fontsize=16)
        axes[i].set_xlim(-0.25, 2.8)

        turn_off_box(axes[i])
        set_tick_size(axes[i], 14)

    fig.tight_layout()
    plt.show()

    # ANOVA for human priority trials
    n_comm = np.asarray(n_comm)
    tf = np.asarray(tf)
    path_len = np.asarray(path_len)

    metrics = [n_comm, tf, path_len]
    metric_names = ["ncomm", "tf", "path_len"]

    for i in range(len(metrics)):
        st, pval = stats.f_oneway(metrics[i][:, 0], metrics[i][:, 1], metrics[i][:, 2])
        print "Statistics for ", metric_names[i], ": (F=", st, ", p=", pval, ")"

        # post-hoc test to find pairwise differences
        groups = np.tile(np.array([1, 2, 3]), (len(metrics[i]), 1))
        tukey = pairwise_tukeyhsd(endog=metrics[i].flatten(), groups=groups.flatten(), alpha=0.05)
        # tukey.plot_simultaneous()
        print tukey.summary()

        # compute p-values
        st_range = np.abs(tukey.meandiffs) / tukey.std_pairs
        print "pvalues are: ", psturng(st_range, len(tukey.groupsunique), tukey.df_total)
        print ">>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<"


if __name__ == "__main__":
    # mixed_exp_stats("/home/yuhang/Documents/proactive_guidance/planner_exp",
    #                 "../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                 0)

    mixed_exp_stats_all("/home/yuhang/Documents/proactive_guidance/planner_exp",
                        "../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
                        [1, 3, 3, 4, 6, 7, 8, 9, 10, 11])

    # users = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    # compute_model_stats("/home/yuhang/Documents/proactive_guidance/training_data",
    #                     users, flag_with_box=True)

    # users = range(11)
    # compute_model_stats("/home/yuhang/Documents/proactive_guidance/training_data",
    #                     users, flag_with_box=True)

    # obs_exp_stats("/home/yuhang/Documents/proactive_guidance/planner_exp_obs",
    #               "../resources/protocols/obs_exp_protocol_3targets_mixed.txt",
    #               "../resources/maps/obs_list_3target.pkl", 0)
