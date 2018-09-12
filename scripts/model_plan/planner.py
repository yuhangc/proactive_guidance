#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import pickle
import time
import threading

from pre_processing import wrap_to_pi
from movement_model import MovementModel, MovementModelParam
from gp_model_approx import GPModelApproxBase
from simulation import Simulator
from policies import MDPFixedTimePolicy
from mcts_policy import MCTSPolicy


class Planner(object):
    def __init__(self):
        # the MCTS planner
        self.mcts_policy = None
        self.human_model = None
        self.modality = None

        self.flag_initialized = False

        # states
        self.s_last = None
        self.s = None
        self.t = 0.0

        self.alp_d_mean = 0.0
        self.alp_d_cov = 0.0

        # (filtered velocities)
        self.v = 0.0
        self.om = 0.0

        self.v_smooth_factor = 0.5
        self.om_smooth_factor = 0.5

        self.cov_m_base = 0.3**2
        self.cov_m_k = 0.3**2 - 0.05**2

    def create_policy(self, default_policy, modality):
        self.mcts_policy = MCTSPolicy(default_policy.tmodel, default_policy, modality)
        self.human_model = default_policy.tmodel
        self.modality = modality

    def reset(self):
        self.s_last = None
        self.s = None
        self.t = 0.0

        self.v = 0.0
        self.om = 0.0

        self.flag_initialized = False

    def compute_plan(self, t_max=0.8):
        if self.flag_initialized:
            # create two threads to run plan separately?
            thread_comm = threading.Thread(target=self.mcts_policy.generate_policy, args=[self.s, t_max-0.05])
            thread_no_comm = threading.Thread(target=self.mcts_policy.generate_policy_no_comm,
                                              args=[self.s, (self.alp_d_mean, self.alp_d_cov**0.5), t_max-0.05])

            thread_comm.start()
            thread_no_comm.start()

            # wait for threads to finish
            thread_comm.join()
            thread_no_comm.join()

            # get and compare plan
            a_opt, v_comm = self.mcts_policy.get_policy()
            v_no_comm = self.mcts_policy.get_policy_no_comm()

            if v_no_comm > v_comm:
                return None
            else:
                return a_opt
        else:
            self.mcts_policy.generate_policy(self.s, t_max)
            a_opt, v_comm = self.mcts_policy.get_policy()

            self.flag_initialized = True

            return a_opt

    def update_state(self, s_new, t_new):
        if self.s is None:
            self.s = s_new.copy()
            self.t = t_new
        else:
            x_inc = s_new[:2] - self.s[:2]
            v_new = np.linalg.norm(x_inc / (t_new - self.t))

            alp_inc = wrap_to_pi(s_new[2] - self.s[2])
            om_new = alp_inc / (t_new - self.t)

            # simple smoothing of velocities
            self.v = self.v_smooth_factor * self.v + (1.0 - self.v_smooth_factor) * v_new
            self.om = self.om_smooth_factor * self.om + (1.0 - self.om_smooth_factor) * om_new

            self.s = s_new.copy()
            self.t = t_new

    def update_alp(self, s):
        # compute measurement
        x_diff = s[:2] - self.s_last[:2]
        alp_m = np.arctan2(x_diff[1], x_diff[0])

        d_inc = np.linalg.norm(x_diff)
        cov_m = max([0.05**2, self.cov_m_base - self.cov_m_k * d_inc])

        K = self.alp_d_cov / (self.alp_d_cov + cov_m)
        self.alp_d_mean += K * wrap_to_pi(alp_m - self.alp_d_mean)
        self.alp_d_mean = wrap_to_pi(self.alp_d_mean)

        self.alp_d_cov = (1.0 - K) * self.alp_d_cov

    def execute_plan(self, s, a):
        self.s_last = s.copy()

        self.alp_d_mean, ad_std = self.human_model.gp_model[self.modality].predict_fast(a)[0]
        self.alp_d_cov = ad_std**2


def validate_planner(policy_path, modality, rep):
    # create a planner
    planner = Planner()

    # load default policies
    with open(policy_path) as f:
        mdp_policy = pickle.load(f)

    human_model = mdp_policy.tmodel

    # load policy into planner
    planner.create_policy(mdp_policy, modality)

    # initial state
    s = np.array([0.5, 0.5, 0.0])
    s_g = mdp_policy.s_g
    t = 0.0
    dt = 0.5
    T = 30.0
    tol = 0.5

    traj_sample_dt = 0.5
    traj_sample_T = 10.0

    t_all = []
    traj_all = []
    comm_states_all = []

    for k in range(rep):
        planner.reset()

        t_list = [0.0]
        traj_list = [s.copy()]
        comm_states = []
        planner.update_state(s, 0.0)
        t_traj = None
        traj = None
        k_traj = 0
        while t < T:
            # check if goal reached
            err = np.linalg.norm(s[:2] - s_g[:2])
            if err <= tol:
                # send stop command to human
                t_traj, traj = human_model.sample_traj_single_action((modality, 10), traj_sample_dt, T-t)
                t_list += list(t_traj)
                traj_list += list(traj)
                t = T
                break

            # compute plan
            a_opt = planner.compute_plan(t_max=0.5)

            if a_opt is None:
                pass
            else:
                comm_states.append(s.copy())

                human_model.set_state(s[0], s[1], s[2])
                t_traj, traj = human_model.sample_traj_single_action((modality, a_opt), traj_sample_dt, traj_sample_T)

                t_list.append(t_traj[1])
                traj_list.append(traj[1].copy())
                t_list.append(t_traj[2])
                traj_list.append(traj[2].copy())

                t = t_traj[2]
                s = traj[2]
                k_traj = 2

                planner.update_state(s, t)
                planner.execute_plan(s, a_opt)

            # go to next state
            k_traj += 1
            t = t_traj[k_traj]
            s = traj[k_traj]

            planner.update_state(s, t)
            planner.update_alp(s)

        t_all.append(np.asarray(t_list))
        traj_all.append(np.asarray(traj_list))
        comm_states_all.append(np.asarray(comm_states))

    # plot
    fig, ax = plt.subplots()
    for traj, comm_states in zip(traj_all, comm_states_all):
        ax.plot(traj[:, 0], traj[:, 1])
        ax.scatter(comm_states[:, 0], comm_states[:, 1])

    plt.show()


if __name__ == "__main__":
    validate_planner("/home/yuhang/Documents/proactive_guidance/training_data/user0/mdp_planenr_obs_haptic.pkl",
                     "haptic", 1)
