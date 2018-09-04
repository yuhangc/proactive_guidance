#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from gp_model_approx import GPModelApproxBase
from movement_model import MovementModel
from policies import NaivePolicy


class Simulator(object):
    def __init__(self):
        # use a simple model first
        self.human_model = MovementModel()

        # temporarily load a preset model
        self.human_model.load_model("/home/yuhang/Documents/proactive_guidance/training_data/user0")
        self.human_model.set_default_param()

        # use a naive planner
        self.planner = NaivePolicy()

        self.dt = 0.5
        self.dt_comm = 0.0

    def run_trial(self, s_init, s_g, modality, T, tol=0.5):
        self.human_model.set_state(s_init[0], s_init[1], s_init[2])

        if modality == "haptic":
            self.dt_comm = 2.0
        else:
            self.dt_comm = 3.0

        t = 0.0
        t_list = ()
        traj_list = ()
        while t < T:
            # check if goal reached
            err = np.linalg.norm(s_init[:2] - s_g[:2])
            if err <= tol:
                # send stop command to human
                t_traj, traj = self.human_model.sample_traj_single_action((modality, 10), self.dt, T-t)
                t_list += t_traj
                traj_list += traj
                t = T
                break

            # compute feedback using planner
            self.planner.set_target(s_g)
            alpha_d = self.planner.compute_policy(self.human_model.s)

            # sample human response giving feedback
            t_traj, traj = self.human_model.sample_traj_single_action((modality, alpha_d), self.dt, self.dt_comm)
            t_traj += t

            t = t_traj[-1]
            s = traj[-1]
            self.human_model.set_state(s[0], s[1], s[2])

            t_list += (t_traj, )
            traj_list += (traj, )

        return np.hstack(t_list), np.vstack(traj_list)


def simulate_naive_policy(n_trials, s_g, modality):
    sim = Simulator()

    traj_list = []
    for i in range(n_trials):
        traj_list.append(sim.run_trial((0.0, 0.0, 0.0), s_g, modality, 30.0, tol=0.5))

    fig, axes = plt.subplots()
    for i in range(n_trials):
        t, traj = traj_list[i]
        axes.plot(traj[:, 0], traj[:, 1])
    axes.axis("equal")

    axes.scatter(s_g[0], s_g[1])

    plt.show()


if __name__ == "__main__":
    simulate_naive_policy(20, np.array([5.0, 2.0, 0.0]), "haptic")
