#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

import pickle

from model_plan.policies import validate_free_space_policy, mkdir_p
from model_plan.simulation import Simulator
from model_plan.movement_model import MovementModel
from model_plan.policies import NaivePolicy, MDPFixedTimePolicy
from model_plan.policies import simulate_naive_policy
from data_processing.loading import wrap_to_pi
from model_plan.plotting_utils import *


def compute_traj_stats(traj_list, s_init, s_goal):
    # reparameterize the trajectories
    traj_list_reparam = []

    d = np.linalg.norm(s_goal[:2] - s_init[:2])
    s = np.linspace(0, d, 50)
    s_vec = (s_goal[:2] - s_init[:2]) / d

    for t, traj in traj_list:
        # append s_goal
        traj = np.vstack((traj, s_goal.reshape(1, -1)))

        s_traj = []
        y_traj = []

        for point in traj:
            u = point[:2] - s_init[:2]
            v = np.dot(u, s_vec) * s_vec
            s_traj.append(np.dot(u, s_vec))

            y = np.linalg.norm(u - v)

            if np.cross(u, v) < 0:
                y_traj.append(y)
            else:
                y_traj.append(-y)

        traj_list_reparam.append(np.interp(s, s_traj, y_traj))

    # compute average trajectory and covariance
    traj_list_reparam = np.asarray(traj_list_reparam)

    traj_avg = np.mean(traj_list_reparam, axis=0)
    traj_std = np.std(traj_list_reparam, axis=0)

    # rotate the thing back
    traj_avg_rotated = []
    traj_ub = []
    traj_lb = []

    th = np.arctan2(s_vec[1], s_vec[0])
    rmat = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for i in range(len(s)):
        point = np.array([s[i], traj_avg[i]])
        point_trans = np.dot(rmat, point) + s_init[:2]
        traj_avg_rotated.append(point_trans)

        point = np.array([s[i], traj_avg[i] + traj_std[i]])
        point_trans = np.dot(rmat, point) + s_init[:2]
        traj_ub.append(point_trans)

        point = np.array([s[i], traj_avg[i] - traj_std[i]])
        point_trans = np.dot(rmat, point) + s_init[:2]
        traj_lb.append(point_trans)

    # return (s, traj_avg, traj_std), (traj_avg_rotated, traj_ub, traj_lb)
    return np.asarray(traj_avg_rotated), np.asarray(traj_ub), np.asarray(traj_lb)


def visualize_policy_traj(protocol_file, usr, policy, s_init, n_rep=30,
                          modality="haptic", style="cov", flag_save=False, flag_unified_policy=False):
    # load the protocol
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    # find out how many targets?
    targets = []
    for trial in protocol_data:
        target = int(trial[0])

        if target not in targets:
            targets.append(target)

    # for each target
    fig, axes = plt.subplots()
    cm = plt.get_cmap("gist_rainbow")

    for target in targets:
        # load policy
        if flag_unified_policy:
            policy_path = "/home/yuhang/Documents/proactive_guidance/training_data/user_unified" + \
                          "/pretrained_model/" + policy + "_" + modality + "/free_space"
        else:
            policy_path = "/home/yuhang/Documents/proactive_guidance/training_data/user" + str(usr) + \
                          "/pretrained_model/" + policy + "_" + modality + "/free_space"
        with open(policy_path + "/target" + str(target) + ".pkl") as f:
            planner = pickle.load(f)

        # create simulator and simulate
        model_path = "/home/yuhang/Documents/proactive_guidance/training_data/user" + str(usr)
        sim = Simulator(planner, model_path)

        traj_list = []
        T = 30.0
        for i in range(n_rep):
            traj_list.append(sim.run_trial(s_init, planner.s_g, modality, T, tol=0.3))

        # compute the average (and covariance?)
        traj_avg, traj_ub, traj_lb = compute_traj_stats(traj_list, s_init, planner.s_g)

        # plot the thing
        if style == "cov":

            axes.plot(traj_ub[:, 0], traj_ub[:, 1], color=(0.7, 0.7, 0.7))
            axes.plot(traj_lb[:, 0], traj_lb[:, 1], color=(0.7, 0.7, 0.7))

            # add a patch
            cov_path = mpath.Path
            path_data = [(cov_path.MOVETO, traj_ub[0, :2])]

            for i in range(1, len(traj_ub)):
                path_data.append((cov_path.LINETO, traj_ub[i, :2]))

            for point in reversed(traj_lb):
                path_data.append((cov_path.LINETO, point[:2]))

            codes, verts = zip(*path_data)
            cov_path = mpath.Path(verts, codes)
            axes.add_patch(mpatches.PathPatch(cov_path,
                                              color=cm(1. * target / len(targets)),
                                              alpha=0.3))

            axes.plot(traj_avg[:, 0], traj_avg[:, 1],
                      color=cm(1. * target / len(targets)),
                      lw=2)

        else:
            axes.plot(traj_avg[:, 0], traj_avg[:, 1],
                      color=cm(1. * target / len(targets)),
                      lw=2)

            # plot all simulated trajectories
            for t, traj in traj_list:
                axes.plot(traj[:, 0], traj[:, 1],
                          color=cm(1. * target / len(targets)),
                          lw=0.5, alpha=0.3)

        axes.axis("equal")

    if flag_save:
        save_path = "/home/yuhang/Documents/proactive_guidance/figures/modeling/user" + str(usr)
        mkdir_p(save_path)
        if flag_unified_policy:
            fig.savefig(save_path + "/unified_policy_" + style + ".jpg")
        else:
            fig.savefig(save_path + "/" + policy + "_policy_" + style + ".jpg")
    else:
        plt.show()


def visualize_obs_policy_traj(usr, policy, n_rep=30, modality="haptic",
                              style="cov", flag_save=False):
    n_targets = 3
    target_all = [[1.5, 4.5], [2.5, 3.5], [3.5, 2]]
    s_init = [-2.0, 0.5, 0.25 * np.pi]

    obs_all = []
    obs_all.append([[0.5, 3.25, 0.5, 2.25], [2.0, 3.25, 0.5, 2.25], [1.0, 5.0, 1.0, 0.5]])
    obs_all.append([[-1.0, 2.5, 2.0, 1.0], [0.0, 1.25, 1.5, 0.5]])
    obs_all.append([[0.0, 0.0, 0.75, 2.25], [0.75, 3.0, 2.5, 0.5], [2.0, 0.0, 1.0, 1.0]])

    # for each target
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5))
    fig.tight_layout()
    cm = plt.get_cmap("gist_rainbow")

    for i in range(n_targets):
        # load policy
        policy_path = "/home/yuhang/Documents/proactive_guidance/training_data/user" + str(usr) + \
                      "/pretrained_model/" + policy + "_" + modality + "/obstacle"
        with open(policy_path + "/target" + str(i) + ".pkl") as f:
            planner = pickle.load(f)

        # create simulator and simulate
        model_path = "/home/yuhang/Documents/proactive_guidance/training_data/user" + str(usr)
        sim = Simulator(planner, model_path)

        traj_list = []
        T = 30.0
        for rep in range(n_rep):
            traj_list.append(sim.run_trial(s_init, planner.s_g, modality, T, tol=0.3))

        # compute the average (and covariance?)
        traj_avg, traj_ub, traj_lb = compute_traj_stats(traj_list, s_init, planner.s_g)

        # plot the thing
        for x, y, w, h in obs_all[i]:
            rect = Rectangle((x, y), w, h)
            axes[i].add_patch(rect)

        # axes[i].scatter(target_all[i][0], target_all[i][1], facecolor='r')
        rect = Rectangle((target_all[i][0], target_all[i][1]), 0.25, 0.25, facecolor='r', lw=0)
        axes[i].add_patch(rect)
        axes[i].scatter(s_init[0], s_init[1])

        axes[i].axis("equal")
        axes[i].set_xlim(-3, 4.5)
        axes[i].set_ylim(-1, 6)

        if style == "cov":

            axes[i].plot(traj_ub[:, 0], traj_ub[:, 1], color=(0.7, 0.7, 0.7))
            axes[i].plot(traj_lb[:, 0], traj_lb[:, 1], color=(0.7, 0.7, 0.7))

            # add a patch
            cov_path = mpath.Path
            path_data = [(cov_path.MOVETO, traj_ub[0, :2])]

            for i in range(1, len(traj_ub)):
                path_data.append((cov_path.LINETO, traj_ub[i, :2]))

            for point in reversed(traj_lb):
                path_data.append((cov_path.LINETO, point[:2]))

            codes, verts = zip(*path_data)
            cov_path = mpath.Path(verts, codes)
            axes[i].add_patch(mpatches.PathPatch(cov_path,
                                                 color='k',
                                                 alpha=0.3))

            axes[i].plot(traj_avg[:, 0], traj_avg[:, 1],
                         color='k',
                         lw=2)

        else:
            axes[i].plot(traj_avg[:, 0], traj_avg[:, 1],
                         color='k',
                         lw=2)

            # plot all simulated trajectories
            for t, traj in traj_list:
                axes[i].plot(traj[:, 0], traj[:, 1],
                             color='k',
                             lw=0.5, alpha=0.3)

    if flag_save:
        save_path = "/home/yuhang/Documents/proactive_guidance/figures/modeling/user" + str(usr)
        mkdir_p(save_path)
        fig.savefig(save_path + "/" + policy + "_policy_obs_" + style + ".jpg")
    else:
        plt.show()


def visualize_policy_diff(p1_file, p2_file):
    # load the policy files
    with open(p1_file) as f:
        p1 = pickle.load(f)

    with open(p2_file) as f:
        p2 = pickle.load(f)

    # range for visualization
    x_range = [-1.5, 4.25]
    y_range = [-1.5, 5.25]
    a_range = [-np.pi, np.pi]
    dx = 0.25
    dy = 0.25

    nX = int((x_range[1] - x_range[0]) / dx) + 1
    nY = int((y_range[1] - y_range[0]) / dy) + 1
    nA = 24

    da = (a_range[1] - a_range[0]) / nA

    policy_diff = np.zeros((nX, nY))

    xgg, ygg, tmp = p2.xy_to_grid(p2.s_g[0], p2.s_g[1], 0.0)

    for xg in range(nX):
        for yg in range(nY):
            if xg == xgg and yg == ygg:
                continue

            if yg == 14 and xg == 19:
                print "hjere"

            diff_max = 0.0
            for ag in range(nA):
            # for ag in [11, 12, 13]:
                x = x_range[0] + dx * xg
                y = y_range[0] + dy * yg
                # a = a_range[0] + da * ag
                a = 0.0

                cmd1 = p1.sample_policy((x, y, a))
                cmd2 = p2.sample_policy((x, y, a))
                cmd_diff = wrap_to_pi(cmd1 - cmd2)

                if np.abs(cmd_diff) > diff_max:
                    diff_max = np.abs(cmd_diff)

            policy_diff[xg, yg] = diff_max

            # for ag in range(nA):
            #     x = x_range[0] + dx * xg
            #     y = y_range[0] + dy * yg
            #     a = a_range[0] + da * ag
            #
            #     cmd1 = p1.sample_policy((x, y, a))
            #     cmd2 = p2.sample_policy((x, y, a))
            #     cmd_diff = wrap_to_pi(cmd1 - cmd2)
            #
            #     diff_max += np.abs(cmd_diff)
            #
            # policy_diff[xg, yg] = diff_max / nA

    policy_diff[xgg, ygg] = np.min(policy_diff)

    print np.max(policy_diff)

    fig, ax = plt.subplots()
    ax.imshow(policy_diff.transpose(), origin="lower", cmap="hot", interpolation="nearest")
    plt.show()


def visualize_naive_mdp_policy_diff(root_path, user, target):
    p1_file = root_path + "/user" + str(user) + \
              "/pretrained_model/naive_haptic/free_space/target" + str(target) + ".pkl"

    p2_file = root_path + "/user" + str(user) + \
              "/pretrained_model/mdp_haptic/free_space/target" + str(target) + ".pkl"

    visualize_policy_diff(p1_file, p2_file)


def align_trial(t, th, dist, t_shift):
    t = np.hstack((np.array([0]), t + t_shift))
    th = np.hstack((np.array([0]), th))
    dist = np.hstack((np.array([0]), dist))

    return t, th, dist


def compute_avg(t_list, x_list, Tmax):
    x_avg = np.zeros((100, ))

    for ti, xi in zip(t_list, x_list):
        t = np.linspace(0, Tmax, 100)
        x = np.interp(t, ti, xi)
        x_avg += x

    t = np.linspace(0, Tmax, 100)

    return t, x_avg / len(t_list)


def compute_std(t_list, x_list, Tmax):
    x_avg = np.zeros((100, ))

    for ti, xi in zip(t_list, x_list):
        t = np.linspace(0, Tmax, 100)
        x = np.interp(t, ti, xi)
        x_avg += x

    t = np.linspace(0, Tmax, 100)
    x_avg = x_avg / len(t_list)

    x_std = np.zeros((100, ))
    for ti, xi in zip(t_list, x_list):
        t = np.linspace(0, Tmax, 100)
        x = np.interp(t, ti, xi)
        x_std += (x - x_avg)**2

    x_std = np.sqrt(x_std / (len(t_list) - 1))

    return t, x_avg, x_std


def visualize_movement_model(root_path, usr, ang):
    modalities = ["haptic", "audio"]

    t_raw = []
    th_raw = []
    dist_raw = []

    for modality in modalities:
        # load the preprocessed data
        path = root_path + "/user" + str(usr) + "/" + modality
        data_file = path + "/raw_transformed.pkl"

        with open(data_file) as f:
            t_all, pose_all, protocol_data = pickle.load(f)

        t_ang = []
        th_ang = []
        dist_ang = []

        for i in range(len(protocol_data)):
            alpha_d = protocol_data[i, 0] - 90.0
            if alpha_d > 180:
                alpha_d -= 360

            if np.abs(alpha_d - ang) < 1e-3:
                t_ang.append(t_all[i] - t_all[i][0])
                th_ang.append(pose_all[i][:, 2])
                dist_ang.append(np.linalg.norm(pose_all[i][:, :2], axis=1))

        t_raw.append(t_ang)
        th_raw.append(th_ang)
        dist_raw.append(dist_ang)

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 4.5))

    # fix some of the alignment issue
    t_raw[0][1], th_raw[0][1], dist_raw[0][1] = align_trial(t_raw[0][1], th_raw[0][1], dist_raw[0][1], 0.9)
    t_raw[0][3], th_raw[0][3], dist_raw[0][3] = align_trial(t_raw[0][3], th_raw[0][3], dist_raw[0][3], 0.9)

    colors = [(0, .298, .569), (.875, .169, 0)]
    T = 5.2
    for i in range(len(modalities)):
        # plot angle changes
        for t, th in zip(t_raw[i], th_raw[i]):
            idx = np.where(t <= T)
            axes[0].plot(t[idx], th[idx], color=colors[i], lw=1.5, alpha=0.3)

        # plot distance changes
        for t, dist in zip(t_raw[i], dist_raw[i]):
            idx = np.where(t <= T)
            axes[1].plot(t[idx], dist[idx], color=colors[i], lw=1.5, alpha=0.3)

    # compute the average
    t, th_avg = compute_avg(t_raw[0], th_raw[0], T)
    p00 = axes[0].plot(t, th_avg, color=colors[0], lw=2.0)
    t, th_avg = compute_avg(t_raw[1], th_raw[1], T)
    p01 = axes[0].plot(t, th_avg, color=colors[1], lw=2.0)

    axes[0].set_xlim(0, 5.2)
    axes[0].set_yticks([0.0, 1.5, 3.0])

    t, th_avg = compute_avg(t_raw[0], dist_raw[0], T)
    p10 = axes[1].plot(t, th_avg, color=colors[0], lw=2.0)
    t, th_avg = compute_avg(t_raw[1], dist_raw[1], T)
    p11 = axes[1].plot(t, th_avg, color=colors[1], lw=2.0)

    axes[1].set_xlim(0, 5.2)
    axes[1].set_yticks([0.0, 0.8, 1.6])

    set_tick_size(axes[0], 14)
    turn_off_box(axes[0])
    set_tick_size(axes[1], 14)
    turn_off_box(axes[1])

    # ax.grid(linestyle="-", color='black', alpha=0.3)

    # axes[0].set_xlabel(xlabel, fontsize=16)
    axes[0].set_ylabel("Heading (rad)", fontsize=16)
    axes[1].set_ylabel("Distance (m)", fontsize=16)
    axes[1].set_xlabel("Time (s)", fontsize=16)

    # axes[0].legend([p00[0], p01[0]], ["Haptic", "Verbal"], loc=0, fancybox=False)
    axes[1].legend([p10[0], p11[0]], ["Haptic", "Verbal"], loc=0, fancybox=False)

    fig.tight_layout()

    # simulate the process
    model = MovementModel()
    model.load_model(root_path + "/user" + str(usr))
    model.set_default_param()

    n_samples = 20
    traj_sim = []
    t_sim = []
    th_sim = []
    dist_sim = []
    for i in range(n_samples):
        traj_sim.append(model.sample_traj_single_action(("haptic", np.deg2rad(ang)), 0.5, T))

    fig1, axes = plt.subplots(2, 1, figsize=(4.5, 4.5))
    colors = [(.875, .169, 0), (0, .298, .569)]

    for t, traj in traj_sim:
        thi = traj[:, 2]
        disti = np.linalg.norm(traj[:, :2], axis=1)

        axes[0].plot(t, thi, color=colors[1], lw=0.75, alpha=0.3)
        p100 = axes[1].plot(t, disti, color=colors[1], lw=0.75, alpha=0.3)

        t_sim.append(t)
        th_sim.append(thi)
        dist_sim.append(disti)

    t, th_avg, th_std = compute_std(t_sim, th_sim, T)
    axes[0].plot(t, th_avg, color=colors[1], lw=2.0)
    # axes[0].fill_between(t, th_avg - th_std, th_avg + th_std,
    #                      alpha=0.2, color='k')
    t, dist_avg, dist_std = compute_std(t_sim, dist_sim, T)
    p101 = axes[1].plot(t, dist_avg, color=colors[1], lw=2.0)
    # axes[1].fill_between(t, dist_avg - dist_std, dist_avg + dist_std,
    #                      alpha=0.2, color='k')

    # plot angle changes
    # for t, th in zip(t_raw[0], th_raw[0]):
    #     idx = np.where(t <= T)
    #     axes[0].plot(t[idx], th[idx], color=colors[0], lw=1.5, alpha=0.3)
    #
    # # plot distance changes
    # for t, dist in zip(t_raw[0], dist_raw[0]):
    #     idx = np.where(t <= T)
    #     axes[1].plot(t[idx], dist[idx], color=colors[0], lw=1.5, alpha=0.3)

    t, th_avg = compute_avg(t_raw[0], th_raw[0], T)
    axes[0].plot(t, th_avg, color=colors[0], lw=2.0)

    axes[0].set_xlim(0, 5.2)
    axes[0].set_yticks([0.0, 1.5, 3.0])

    t, th_avg = compute_avg(t_raw[0], dist_raw[0], T)
    p11 = axes[1].plot(t, th_avg, color=colors[0], lw=2.0)

    axes[1].set_xlim(0, 5.2)
    axes[1].set_yticks([0.0, 0.8, 1.6])

    set_tick_size(axes[0], 14)
    turn_off_box(axes[0])
    set_tick_size(axes[1], 14)
    turn_off_box(axes[1])

    # ax.grid(linestyle="-", color='black', alpha=0.3)

    # axes[0].set_xlabel(xlabel, fontsize=16)
    axes[0].set_ylabel("Heading (rad)", fontsize=16)
    axes[1].set_ylabel("Distance (m)", fontsize=16)
    axes[1].set_xlabel("Time (s)", fontsize=16)

    axes[1].legend([p101[0], p11[0]], ["Simulation", "Measurement"], loc=0, fancybox=False)

    fig1.tight_layout()

    plt.show()


if __name__ == "__main__":
    # s_init = np.array([-1.0, 2.0, 0.0])
    # visualize_policy_traj("../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                       1, "naive", s_init, flag_save=True, style="sample")

    # visualize_policy_traj("../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                       10, "mdp", s_init, flag_save=True, flag_unified_policy=True)

    # simulate_naive_policy(30, np.array([2.46, 4.00, 0.0]), "haptic", 3)

    # visualize_naive_mdp_policy_diff("/home/yuhang/Documents/proactive_guidance/training_data", 4, 0)

    # visualize_obs_policy_traj(3, "naive", flag_save=True, style="sample")

    visualize_movement_model("/home/yuhang/Documents/proactive_guidance/training_data", 0, 105)
