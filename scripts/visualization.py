#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle

from model_plan.policies import validate_free_space_policy
from model_plan.simulation import Simulator


def compute_traj_stats(traj_list, s_init, s_goal):
    # reparameterize the trajectories
    traj_list_reparam = []

    d = np.linalg.norm(s_goal[:2] - s_init[:2])
    s = np.linspace(0, d, 50)
    s_vec = (s_goal[:2] - s_init[:2]) / d

    for t, traj in traj_list:
        # append s_goal
        traj.append(s_goal.copy())

        s_traj = []
        y_traj = []

        for point in traj:
            u = point[:2] - s_init[:2]
            v = np.dot(u, s_vec) * s_vec
            s_traj.append(np.dot(u, s_vec))

            y = np.linalg.norm(u - v)

            if np.cross(u, v) > 0:
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
        point_trans = np.dot(rmat, point) + s_init



    return s, traj_avg, traj_std


def visualize_mdp_policy_traj(protocol_file, usr, policy, s_init, n_rep=30, modality="haptic", style="cov"):
    # load the protocol
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    # find out how many targets?
    targets = []
    for trial in protocol_data:
        target = int(trial[0])

        if target not in targets:
            targets.append(target)

    # for each target
    for target in targets:
        # load policy
        policy_path = "/home/yuhang/Documents/proactive_guidance/training_data/user" + str(usr) + \
                      "/pretrained_mode/" + policy + "_" + modality + "/free_space"
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
        s, traj_avg, traj_std = compute_traj_stats(traj_list, s_init, planner.s_g)

        # plot the thing


if __name__ == "__main__":
    pass
