#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

import pickle

from model_plan.policies import validate_free_space_policy, mkdir_p
from model_plan.simulation import Simulator
from model_plan.policies import NaivePolicy, MDPFixedTimePolicy
from model_plan.policies import simulate_naive_policy
from data_processing.loading import wrap_to_pi


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


if __name__ == "__main__":
    s_init = np.array([-1.0, 2.0, 0.0])
    visualize_policy_traj("../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
                          1, "naive", s_init, flag_save=True, style="sample")

    # visualize_policy_traj("../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                       10, "mdp", s_init, flag_save=True, flag_unified_policy=True)

    # simulate_naive_policy(30, np.array([2.46, 4.00, 0.0]), "haptic", 3)

    # visualize_naive_mdp_policy_diff("/home/yuhang/Documents/proactive_guidance/training_data", 4, 0)
