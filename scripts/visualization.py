#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Circle
import matplotlib.gridspec as gridspec

import pickle

from scipy.signal import savgol_filter

from model_plan.policies import validate_free_space_policy, mkdir_p
from model_plan.simulation import Simulator
from model_plan.movement_model import MovementModel
from model_plan.policies import NaivePolicy, MDPFixedTimePolicy
from model_plan.policies import simulate_naive_policy
from model_plan.gp_model_approx import GPModelApproxBase
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

    c = 0
    colors = [[(.161, .267, .49), (.133, .412, .412)], [(.729, .18, .196), (.733, .431, .18)]]
    color_id = [0, 1, 1, 0, 0, 1, 1]

    # for each target
    fig, axes = plt.subplots(figsize=(3.75, 4))

    for target in targets:
        # load policy
        if flag_unified_policy:
            policy_path = "/home/yuhang/Documents/proactive_guidance/training_data/unified" + \
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
            traj_list.append(sim.run_trial(s_init, planner.s_g, modality, T, tol=0.35))

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
                                              color=colors[c][color_id[target]],
                                              alpha=0.3))

            axes.plot(traj_avg[:, 0], traj_avg[:, 1],
                      color=colors[c][color_id[target]],
                      lw=2)

        else:
            axes.plot(traj_avg[:, 0], traj_avg[:, 1],
                      color=colors[c][color_id[target]],
                      lw=2)

            # plot all simulated trajectories
            for t, traj in traj_list:
                axes.plot(traj[:, 0], traj[:, 1],
                          color=colors[c][color_id[target]],
                          lw=0.5, alpha=0.3)

        circ = Circle((planner.s_g[0], planner.s_g[1]),
                      radius=0.35, fill=False, alpha=0.5, lw=1, linestyle='--')
        axes.add_patch(circ)

    axes.axis("equal")
    axes.set_title("Personalized Model", fontsize=16)
    axes.set_ylim(-1, 5)
    set_tick_size(axes, 14)

    fig.tight_layout()
    axes.set_ylim(-1, 5)
    axes.set_xlim(-1.25, 4.3)

    if flag_save:
        save_path = "/home/yuhang/Documents/proactive_guidance/figures/modeling/user" + str(usr)
        mkdir_p(save_path)
        if flag_unified_policy:
            fig.savefig(save_path + "/unified_policy_" + style + ".jpg")
        else:
            fig.savefig(save_path + "/" + policy + "_policy_" + style + ".jpg")
    else:
        plt.show()


def visualize_policies_free_space(usr, s_init):
    policies = ["mdp", "naive"]
    ntargets = 7
    T = 30.0
    n_rep = 30

    model_path = "/home/yuhang/Documents/proactive_guidance/training_data/user" + str(usr)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))

    colors = [[(.161, .267, .49), (.133, .412, .412)], [(.729, .18, .196), (.733, .431, .18)]]
    color_id = [0, 1, 1, 0, 0, 1, 1]

    titles = ["Optimized Policy", "Naive Policy"]

    for ip, policy in enumerate(policies):
        policy_path = "/home/yuhang/Documents/proactive_guidance/training_data/user" + str(usr) + \
                      "/pretrained_model/" + policy + "_haptic/free_space"

        for target in range(ntargets):
            with open(policy_path + "/target" + str(target) + ".pkl") as f:
                planner = pickle.load(f)

            # create simulator and simulate
            sim = Simulator(planner, model_path)

            traj_list = []

            for i in range(n_rep):
                traj_list.append(sim.run_trial(s_init, planner.s_g, "haptic", T, tol=0.3))

            # compute the average (and covariance?)
            traj_avg, traj_ub, traj_lb = compute_traj_stats(traj_list, s_init, planner.s_g)

            axes[ip].plot(traj_avg[:, 0], traj_avg[:, 1],
                          color=colors[ip][color_id[target]],
                          lw=2)

            # plot all simulated trajectories
            for t, traj in traj_list:
                axes[ip].plot(traj[:, 0], traj[:, 1],
                              color=colors[ip][color_id[target]],
                              lw=0.75, alpha=0.2)

            # plot the target?
            circ = Circle((planner.s_g[0], planner.s_g[1]),
                          radius=0.35, fill=False, alpha=0.5, lw=1, linestyle='--')
            axes[ip].add_patch(circ)

        axes[ip].axis("equal")
        axes[ip].set_title(titles[ip])
        axes[ip].set_ylim(-1, 5)
        set_tick_size(axes[ip], 14)

    fig.tight_layout()
    axes[0].set_ylim(-1, 5)
    axes[0].set_xlim(-1.25, 4.3)
    # axes[0].grid()
    axes[1].set_ylim(-1, 5)
    axes[1].set_xlim(-1.25, 4.3)
    # axes[1].grid()

    plt.show()


def visualize_obs_policy_traj(usr, target, n_rep=30, modality="haptic",
                              style="cov", flag_save=False):
    map_file = "../resources/maps/obs_list_3target.pkl"
    with open(map_file) as f:
        target_all, obs_all = pickle.load(f)

    s_init = [-2.0, 0.5, 0.25 * np.pi]

    # for each target
    n_policy = 2
    policies = ["naive", "mdp"]
    policy_names = ["Naive Policy", "Optimized Policy"]

    colors = [(0., 0., 0.), (.161, .267, .49)]

    fig, axes = plt.subplots(1, n_policy, figsize=(8, 3.5))

    for i in range(n_policy):
        # load policy
        policy_path = "/home/yuhang/Documents/proactive_guidance/training_data/user" + str(usr) + \
                      "/pretrained_model/" + policies[i] + "_" + modality + "/obstacle"
        with open(policy_path + "/target" + str(target) + ".pkl") as f:
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
        for x, y, w, h in obs_all[target]:
            rect = Rectangle((x, y), w, h, facecolor=(0.7, 0.7, 0.7), hatch='x', lw=2, edgecolor=(0.3, 0.3, 0.3))
            axes[i].add_patch(rect)

        # axes[i].scatter(target_all[i][0], target_all[i][1], facecolor='r')
        circ = Circle((target_all[target][0] + 0.125, target_all[target][1] + 0.125),
                      radius=0.35, facecolor=(0.8, 0.0, 0.0), alpha=0.15, lw=1, linestyle='--', edgecolor='k')
        axes[i].add_patch(circ)

        axes[i].axis("equal")
        axes[i].set_xlim(-2.5, 3.5)
        axes[i].set_ylim(-0.3, 4.5)
        axes[i].set_title(policy_names[i], fontsize=16)

        for tick in axes[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in axes[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

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
                         color=colors[i],
                         lw=0)

            # plot all simulated trajectories
            for t, traj in traj_list:
                axes[i].plot(traj[:, 0], traj[:, 1],
                             color=colors[i],
                             lw=0.0, alpha=0.3)

    fig.tight_layout()

    if flag_save:
        save_path = "/home/yuhang/Documents/proactive_guidance/figures/modeling/user" + str(usr)
        mkdir_p(save_path)
        fig.savefig(save_path + "/obs_target_" + str(target) + "_" + style + ".jpg")
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


def visualize_all_models(root_path, modality, users):
    # first load the unified model
    with open(root_path + "/unified/gp_model_" + modality + ".pkl") as f:
        model = pickle.load(f)

    n_samples = 50
    x = np.linspace(-np.pi, np.pi, n_samples)

    y_mean = model.predict(x.reshape(n_samples, -1))
    y = y_mean[0][0]

    fig, axes = plt.subplots(figsize=(4, 4))
    axes.plot(x, y, 'k', lw=2)

    for user in users:
        with open(root_path + "/user" + str(user) + "/gp_model_" + modality + ".pkl") as f:
            model = pickle.load(f)

        n_samples = 50
        x = np.linspace(-np.pi, np.pi, n_samples)

        y_mean = model.predict(x.reshape(n_samples, -1))
        y = y_mean[0][0]
        axes.plot(x, y, 'k', lw=1, alpha=0.3)

    axes.axis("equal")
    plt.show()


def visualize_traj_and_feedback(root_path, protocol_file, traj_ids):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_trial = len(protocol_data)

    # generate a color map
    n_colors = int(np.max(protocol_data[:, 0])) + 1
    # cm = plt.get_cmap("gist_rainbow")
    cm = plt.get_cmap("viridis")

    n_cond = 3
    n_target = 7
    cond_name = ["Naive Policy", "Optimized Policy", "Communicate As Needed"]
    # pose_all = [[] for i in range(n_cond)]

    fig, axes = plt.subplots(1, n_cond, figsize=(12, 4.5))

    # plot the goals
    target_pos = np.zeros((n_target, 2))
    visited = np.zeros((100, ))

    for trial in range(n_trial):
        trial_id = int(protocol_data[trial, 0])
        if visited[trial_id] < 1.0:
            visited[trial_id] = 1.0
            target_pos[trial_id] = protocol_data[trial, 2:4]
            for i in range(n_cond):
                # circ = Circle((protocol_data[trial, 2], protocol_data[trial, 3]),
                #               radius=0.35, fill=False, alpha=0.5, lw=1, linestyle='--')
                circ = Circle((protocol_data[trial, 2], protocol_data[trial, 3]),
                              radius=0.35, facecolor=cm(1. * trial_id / n_colors), fill=True,
                              alpha=0.5, lw=1, linestyle='--', edgecolor='k')
                axes[i].add_patch(circ)

    with open(root_path + "/traj_raw.pkl") as f:
        traj_data = pickle.load(f)

    with open(root_path + "/comm_raw.pkl") as f:
        comm_data = pickle.load(f)

    for cond in range(n_cond):
        for target in range(n_target):
            for i, traj in enumerate(traj_data[cond][target]):
                if i == traj_ids[cond][target]:
                    # extend the traj a little for visual effect
                    s = traj[-1].copy()
                    s_g = target_pos[target]

                    d = np.linalg.norm(s[1:3] - s_g)
                    if d >= 0.3:
                        th = np.arctan2(s_g[1] - s[2], s_g[0] - s[1])
                        s += np.array([1.0, np.cos(th), np.sin(th), 0.0, 0.0, 0.0]) * 0.1
                        traj = np.vstack((traj, s.reshape(1, -1)))

                    # smooth the trajectory
                    traj_smooth = savgol_filter(traj, 41, 3, axis=0)

                    # align the communication with the trajectory
                    t_traj = traj[:, 0]
                    comm = comm_data[cond][target][i].reshape(-1, 2)
                    t_comm = comm[:, 0]

                    t_traj -= t_traj[0]
                    t_comm -= t_comm[0]

                    for t in range(len(t_comm)):
                        x = np.interp(t_comm[t], t_traj, traj[:, 1])
                        y = np.interp(t_comm[t], t_traj, traj[:, 2])
                        th = np.deg2rad(np.interp(t_comm[t], t_traj, traj[:, 3]))

                        # plot the arrow
                        dl = 0.3
                        th += comm[t, 1]

                        axes[cond].arrow(x, y, dl * np.cos(th), dl * np.sin(th), alpha=0.6,
                                         head_width=0.05, head_length=0.1)

                    axes[cond].plot(traj_smooth[:, 1], traj_smooth[:, 2],
                                    color=cm(1. * target / n_colors), lw=1.0, alpha=0.6)

    for i in range(n_cond):
        axes[i].set_title(cond_name[i], fontsize=16)

    fig.tight_layout()

    for i in range(n_cond):
        axes[i].axis("equal")
        axes[i].set_title(cond_name[i], fontsize=16)
        axes[i].set_xlim(-1.5, 4.5)
        axes[i].set_ylim(-1.5, 5.5)

        for tick in axes[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in axes[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

    plt.show()


def visualize_err_vs_control(root_path, protocol_file, target, trial_id):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_cond = 3
    cond_name = ["Naive Policy", "Optimized Policy", "Communicate As Needed"]
    # pose_all = [[] for i in range(n_cond)]

    n_colors = 7
    # cm = plt.get_cmap("gist_rainbow")
    cm = plt.get_cmap("viridis")

    axes = [[] for i in range(3)]
    spans = [2, 1, 1]
    ypos = [0, 2, 3]

    fig = plt.figure(figsize=(8, 5.5))
    for i in range(3):
        for j in range(n_cond):
            axes[i].append(plt.subplot2grid((max(ypos)+1, n_cond), (ypos[i], j), rowspan=spans[i]))

    # get the goal position
    target_pos = np.zeros((2, ))

    for data in protocol_data:
        if data[0] == target:
            target_pos = data[2:4]
            break

    with open(root_path + "/traj_raw.pkl") as f:
        traj_data = pickle.load(f)

    with open(root_path + "/comm_raw.pkl") as f:
        comm_data = pickle.load(f)

    for cond in range(n_cond):
        traj = traj_data[cond][target][trial_id[cond]]
        comm = comm_data[cond][target][trial_id[cond]]

        t_traj = traj[:, 0]
        t_comm = comm[:, 0]

        t_traj -= t_traj[0]
        t_comm -= t_comm[0]

        # compute heading error over time
        x_diff = traj[:, 1] - target_pos[0]
        y_diff = traj[:, 2] - target_pos[1]

        th_desired = np.arctan2(-y_diff, -x_diff)
        th_diff = wrap_to_pi(th_desired - np.deg2rad(traj[:, 3]))

        th_diff = savgol_filter(th_diff, 41, 3, axis=0)

        # plot the trajectory
        # traj = savgol_filter(traj, 41, 3, axis=0)
        axes[0][cond].plot(traj[:, 1], traj[:, 2], color='k', alpha=0.5)

        # target
        circ = Circle((target_pos[0], target_pos[1]),
                      radius=0.35, facecolor=cm(5. / n_colors), fill=True,
                      alpha=0.5, lw=1, linestyle='--', edgecolor='k')
        axes[0][cond].add_patch(circ)
        axes[0][cond].axis('equal')

        for t in range(len(t_comm)):
            x = np.interp(t_comm[t], t_traj, traj[:, 1])
            y = np.interp(t_comm[t], t_traj, traj[:, 2])
            th = np.deg2rad(np.interp(t_comm[t], t_traj, traj[:, 3]))

            # plot the arrow
            dl = 0.3
            th += comm[t, 1]

            axes[0][cond].arrow(x, y, dl * np.cos(th), dl * np.sin(th),
                                head_width=0.08, head_length=0.1,
                                edgecolor='k', facecolor='k')

        axes[0][cond].set_title(cond_name[cond], fontsize=12)
        axes[0][cond].set_xlim(-1.5, 2.5)

        # plot
        p0 = axes[1][cond].plot(t_traj, th_diff, '-', color=(.173, .298, .475), linewidth=2.0)
        # axes[cond].stem(t_comm, comm[:, 1], '-.')

        p10 = axes[1][cond].scatter(t_comm, comm[:, 1], facecolors='none', edgecolors=(.722, .22, .22), linewidth=2.0)

        t_comm = np.hstack((t_comm, np.array([t_traj[-1]])))
        comm_plot = np.hstack((np.array([comm[0, 1]]), comm[:, 1]))
        p11 = axes[1][cond].step(t_comm, comm_plot, color=(.722, .22, .22), alpha=0.5)

        # plot a zero line
        axes[1][cond].plot(np.array([0, t_traj[-1]]), np.array([0, 0]), '-.k', linewidth=1.0)
        axes[1][cond].set_xlim(0, t_traj[-1])

        if cond == 0:
            axes[1][cond].set_ylabel("    (rad)", fontsize=12)
            axes[1][cond].legend([p0[0], (p10, p11[0])], ["Error", "Guidance"],
                                 loc=0, fancybox=False, bbox_to_anchor=(0.04, 0.5), frameon=False)

        turn_off_box(axes[1][cond])

        # plot the distance over time
        d_diff = np.sqrt(x_diff**2 + y_diff**2)
        # d_diff = savgol_filter(d_diff, 41, 3)
        axes[2][cond].plot(t_traj, d_diff, color=(.173, .298, .475), linewidth=2.0)
        axes[2][cond].plot(np.array([0, t_traj[-1]]), np.array([0.35, 0.35]), '-.k', linewidth=1.0)
        axes[2][cond].set_xlim(0, t_traj[-1])

        if cond == 0:
            axes[2][cond].set_ylabel("    (m)", fontsize=12)
        axes[2][cond].set_xlabel("Time (s)", fontsize=12)
        turn_off_box(axes[2][cond])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    s_init = np.array([-1.0, 2.0, 0.0])
    # visualize_policy_traj("../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                       3, "naive", s_init, flag_save=True, style="sample")
    # visualize_policies_free_space(3, s_init)

    # visualize_policy_traj("../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                       4, "mdp", s_init, style="sample", flag_unified_policy=False)

    # simulate_naive_policy(30, np.array([2.46, 4.00, 0.0]), "haptic", 3)

    # visualize_naive_mdp_policy_diff("/home/yuhang/Documents/proactive_guidance/training_data", 4, 0)

    # visualize_obs_policy_traj(0, 1, flag_save=False, style="sample")

    # visualize_movement_model("/home/yuhang/Documents/proactive_guidance/training_data", 0, 105)

    # users = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    # # users = [9]
    # visualize_all_models("/home/yuhang/Documents/proactive_guidance/training_data", "haptic", users)

    # traj_ids = [[0, 0, 0, 0, 0, 0, 0],
    #             [0, 0, 0, 0, 0, 2, 0],
    #             [0, 2, 2, 0, 0, 0, 1]]
    # visualize_traj_and_feedback("/home/yuhang/Documents/proactive_guidance/planner_exp/user3",
    #                             "../resources/protocols/free_space_exp_protocol_7targets_mixed.txt",
    #                             traj_ids)

    trial_ids = [1, 2, 0]
    visualize_err_vs_control("/home/yuhang/Documents/proactive_guidance/planner_exp/user3",
                             "../resources/protocols/free_space_exp_protocol_7targets_mixed.txt",
                             5, trial_ids)
