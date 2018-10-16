#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from gp_model_approx import GPModelApproxBase
from movement_model import MovementModel


class Simulator(object):
    def __init__(self, planner, model_path=None):
        # use a simple model first
        self.human_model = MovementModel()

        # temporarily load a preset model
        if model_path is None:
            self.human_model.load_model("/home/yuhang/Documents/proactive_guidance/training_data/user0")
        else:
            self.human_model.load_model(model_path)

        self.human_model.set_default_param()

        self.planner = planner

        self.dt = 0.5
        self.dt_comm = 0.0

    def check_stop(self, traj, s_g, tol):
        for i in range(len(traj)):
            err = np.linalg.norm(traj[i, :2] - s_g[:2])
            if err < tol:
                return i, True

        return len(traj), False

    def run_trial(self, s_init, s_g, modality, T, tol=0.3):
        self.human_model.set_state(s_init[0], s_init[1], s_init[2])

        if modality == "haptic":
            self.dt_comm = 2.0
        else:
            self.dt_comm = 3.0

        # compute policy once
        self.planner.compute_policy(s_g, modality)

        t = 0.0
        t_list = ()
        traj_list = ()
        while t < T:
            # compute feedback using planner
            alpha_d = self.planner.sample_policy(self.human_model.s)

            # sample human response giving feedback
            if t == 0.0:
                t_traj, traj = self.human_model.sample_traj_single_action((modality, alpha_d), self.dt, self.dt_comm,
                                                                          flag_delay_move=False)
            else:
                t_traj, traj = self.human_model.sample_traj_single_action((modality, alpha_d), self.dt, self.dt_comm)
            t_traj += t

            t = t_traj[-1]
            s = traj[-1]
            self.human_model.set_state(s[0], s[1], s[2])

            # check if goal reached
            tf, flag_stopped = self.check_stop(traj, s_g, tol)
            if flag_stopped:
                t_list += (t_traj[:tf], )
                traj_list += (traj[:tf], )
                break

            t_list += (t_traj, )
            traj_list += (traj, )

            # check if goal reached
            # err = np.linalg.norm(s[:2] - s_g[:2])
            # if err <= tol:
            #     # send stop command to human
            #     # t_traj, traj = self.human_model.sample_traj_single_action((modality, 10), self.dt, T-t)
            #     # t_list += t_traj
            #     # traj_list += traj
            #     # t = T
            #     break

        return np.hstack(t_list), np.vstack(traj_list)


if __name__ == "__main__":
    pass
