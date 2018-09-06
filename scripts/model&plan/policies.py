#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import deque

from pre_processing import wrap_to_pi
from movement_model import MovementModel
from gp_model_approx import GPModelApproxBase
from simulation import Simulator


class NaivePolicy(object):
    def __init__(self):
        self.s_g = None

    def compute_policy(self, s_g, modality, max_iter=0):
        self.s_g = s_g

    def sample_policy(self, s):
        # directly point to the direction of goal
        alpha_d = np.arctan2(self.s_g[1] - s[1], self.s_g[0] - s[0])
        alpha_d = wrap_to_pi(alpha_d - s[2])

        return alpha_d


class MDPFixedTimePolicy(object):
    def __init__(self, tmodel):
        # workspace size, resolution, offset
        self.nX = 20
        self.nY = 20
        self.nAlp = 12

        self.dx = 0.25
        self.dy = 0.25
        self.dalp = 2.0 * np.pi / self.nAlp

        self.x_offset = 0.0
        self.y_offset = 0.
        self.alp_offset = -np.pi

        self.x_range = [self.x_offset, self.x_offset + self.dx * (self.nX-1)]
        self.y_range = [self.y_offset, self.y_offset + self.dy * (self.nY-1)]

        # fixed time interval
        self.dt = 2.0

        # number of actions
        self.nA = 11
        self.a_list = np.arange(0, self.nA) * (np.pi / self.nA) - np.pi

        # value function, Q function, policy, rewards
        self.V = -1000.0 * np.ones((self.nX, self.nY, self.nAlp))
        self.Q = np.zeros((self.nX, self.nY, self.nAlp, self.nA))
        self.policy = np.zeros((self.nX, self.nY, self.nAlp))

        self.obs = np.zeros((self.nX, self.nY))

        self.r_goal = 10.0
        self.r_obs = -20.0

        # discount factor
        self.gamma = 0.95

        # policy update counting threshold
        self.n_update_th = 3

        # how many samples to draw for each update
        self.n_samples = 10

        # transition model for sampling
        self.tmodel = tmodel

        # to prevent repeatedly compute policy for same goal
        self.s_g = None
        self.modality = None
        self.flag_policy_computed = False

        # list that stores the update sequence
        self.update_q = []

    # helper functions
    def xy_to_grid(self, x, y, alp):
        xg = int((x - self.x_offset) / self.dx)
        yg = int((y - self.y_offset) / self.dy)
        ag = int((alp - self.alp_offset) / self.dalp)

        return xg, yg, ag

    def grid_to_xy(self, xg, yg, ag):
        x = xg * self.dx + self.x_offset
        y = yg * self.dy + self.y_offset
        alp = ag * self.dalp + self.alp_offset

        return x, y, alp

    def get_value(self, f, x, y, alp):
        xf = (x - self.x_offset) / self.dx
        yf = (y - self.y_offset) / self.dy
        af = (alp - self.alp_offset) / self.dalp
        xg = int(xf)
        yg = int(yf)
        ag = int(af)
        ag1 = (ag + 1) % self.nAlp

        xf -= xg
        yf -= yg
        af -= ag

        if xg + 1 >= self.nX or yg + 1 >= self.nY:
            print "xxx"
            return self.r_obs

        f1 = (1.0 - xf) * f[xg, yg, ag] + xf * f[xg+1, yg, ag]
        f2 = (1.0 - xf) * f[xg, yg+1, ag] + xf * f[xg+1, yg+1, ag]
        g1 = (1.0 - yf) * f1 + yf * f2

        f1 = (1.0 - xf) * f[xg, yg, ag1] + xf * f[xg+1, yg, ag1]
        f2 = (1.0 - xf) * f[xg, yg+1, ag1] + xf * f[xg+1, yg+1, ag1]
        g2 = (1.0 - yf) * f1 + yf * f2

        return (1.0 - af) * g1 + af * g2

    def get_value_ang(self, f, x, y, alp):
        xf = (x - self.x_offset) / self.dx
        yf = (y - self.y_offset) / self.dy
        af = (alp - self.alp_offset) / self.dalp
        xg = int(xf)
        yg = int(yf)
        ag = int(af)
        ag1 = (ag + 1) % self.nAlp

        xf -= xg
        yf -= yg
        af -= ag

        if xg + 1 >= self.nX or yg + 1 >= self.nY:
            print "xxx"
            return self.r_obs

        f1 = f[xg, yg, ag] + xf * wrap_to_pi(f[xg+1, yg, ag] - f[xg, yg, ag])
        f2 = f[xg, yg+1, ag] + xf * wrap_to_pi(f[xg+1, yg+1, ag] - f[xg, yg+1, ag])
        g1 = f1 + yf * wrap_to_pi(f2 - f1)

        f1 = f[xg, yg, ag1] + xf * wrap_to_pi(f[xg+1, yg, ag1] - f[xg, yg, ag1])
        f2 = f[xg, yg+1, ag1] + xf * wrap_to_pi(f[xg+1, yg+1, ag1] - f[xg, yg+1, ag1])
        g2 = f1 + yf * wrap_to_pi(f2 - f1)

        return g1 + af * wrap_to_pi(g2 - g1)

    def load_env(self, file_path):
        # load possible obstacles
        pass

    def init_value_function(self, s_g):
        # set obstacle values
        for xg in range(self.nX):
            for yg in range(self.nY):
                for alp_g in range(self.nAlp):
                    self.V[xg, yg, alp_g] = self.r_obs * self.obs[xg, yg]

        # set goal value
        xgg, ygg, tmp = self.xy_to_grid(s_g[0], s_g[1], 0.0)
        self.V[xgg, ygg] += self.r_goal

        # perform a BFS to initialize values of all states and obtain an update sequence
        visited = np.zeros((self.nX, self.nY))
        nodes = deque()

        visited[xgg, ygg] = 1.0
        nodes.append((xgg, ygg))

        dx = [0, 1, 1, 1, 0, -1, -1, -1]
        dy = [1, 1, 0, -1, -1, -1, 0, 1]
        while nodes:
            xg, yg = nodes.popleft()

            for i in range(8):
                xg_next = xg + dx[i]
                yg_next = yg + dy[i]

                if xg_next < 0 or yg_next < 0 or xg_next >= self.nX or yg_next >= self.nY:
                    continue
                if self.obs[xg_next, yg_next] > 0:
                    continue

                if not visited[xg_next, yg_next]:
                    visited[xg_next, yg_next] = True
                    nodes.append((xg_next, yg_next))
                    self.update_q.append((xg_next, yg_next))

                    # update value and initial policy
                    for alp_g in range(self.nAlp):
                        # value is same for all orientation
                        self.V[xg_next, yg_next, alp_g] = self.gamma * self.V[xg, yg, 0]

                        # apply naive policy
                        x, y, alp = self.grid_to_xy(xg_next, yg_next, alp_g)
                        self.policy[xg_next, yg_next, alp_g] = wrap_to_pi(np.arctan2(-dy[i], -dx[i]) - alp)

        self.visualize_policy()
        plt.show()

    def compute_policy(self, s_g, modality, max_iter=50):
        if self.flag_policy_computed and self.modality == modality:
            goal_diff = np.linalg.norm(self.s_g[:2] - s_g[:2])
            if goal_diff < 0.5:
                print "No need to recompute policy"
                return

        self.s_g = s_g
        self.modality = modality

        if modality == "haptic":
            self.dt = 2.0
        else:
            self.dt = 3.0

        self.init_value_function(s_g)
        xgg, ygg, tmp = self.xy_to_grid(s_g[0], s_g[1], 0.0)

        counter_policy_not_updated = 0
        for i in range(max_iter):
            print "At iteration ", i, "..."

            flag_policy_update = False
            # V_curr = self.V
            # self.V = np.zeros((self.nX, self.nY, self.nAlp))

            # iterate over all states
            for xg, yg in self.update_q:
                for alp_g in range(self.nAlp):
                    # don't perform update on obstacle and goal state
                    if self.obs[xg, yg] > 0:
                        continue
                    if xg == xgg and yg == ygg:
                        continue

                    x, y, alp = self.grid_to_xy(xg, yg, alp_g)

                    # iterate over all actions
                    Q_max = -1000.0
                    a_opt = 0.0

                    # only sample actions that make sense
                    da = np.pi / 3.0 / self.nA
                    a_list = da * np.arange(0, self.nA) - np.pi / 6.0 + self.policy[xg, yg, alp_g]

                    for ai, a in enumerate(a_list):
                        Vnext = 0.0
                        for k in range(self.n_samples):
                            # sample a new state
                            self.tmodel.set_state(x, y, alp)
                            s_next = self.tmodel.sample_state((modality, a), 0.5, self.dt)
                            if s_next[0] < self.x_range[0] or s_next[0] >= self.x_range[1] or \
                                            s_next[1] < self.y_range[0] or s_next[1] >= self.y_range[1]:
                                Vnext += self.r_obs
                            else:
                                Vnext += self.get_value(self.V, s_next[0], s_next[1], s_next[2])

                        # if Vnext != 0:
                        #     print "here"

                        self.Q[xg, yg, alp_g, ai] = self.gamma * Vnext / self.n_samples

                        if self.Q[xg, yg, alp_g, ai] > Q_max:
                            Q_max = self.Q[xg, yg, alp_g, ai]
                            a_opt = a

                    # update value function and policy
                    self.V[xg, yg, alp_g] = Q_max
                    if self.policy[xg, yg, alp_g] != a_opt:
                        self.policy[xg, yg, alp_g] = a_opt
                        flag_policy_update = True

            if not flag_policy_update:
                counter_policy_not_updated += 1
                print "Policy not updated in ", counter_policy_not_updated, "iterations"
                if counter_policy_not_updated >= self.n_update_th:
                    print "Policy converged!"
                    break
            else:
                counter_policy_not_updated = 0

            # self.visualize_policy()
            # plt.show()

        self.flag_policy_computed = True

    def sample_policy(self, s):
        return self.get_value_ang(self.policy, s[0], s[1], wrap_to_pi(s[2]))

    def visualize_policy(self, ax=None):
        Vplot = np.max(self.V, axis=2)
        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(Vplot.transpose(), origin="lower", cmap="hot", interpolation="nearest")


def simulate_naive_policy(n_trials, s_g, modality):
    planner = NaivePolicy()

    sim = Simulator(planner)

    traj_list = []
    for i in range(n_trials):
        traj_list.append(sim.run_trial((0.0, 0.0, 0.0), s_g, modality, 15.0, tol=0.5))

    fig, axes = plt.subplots()
    for i in range(n_trials):
        t, traj = traj_list[i]
        axes.plot(traj[:, 0], traj[:, 1])
    axes.axis("equal")

    axes.scatter(s_g[0], s_g[1])

    plt.show()


def validate_MDP_policy(root_path, flag_plan=True):
    s_g = np.array([3.0, 2.0, 0.0])
    modality = "haptic"
    if flag_plan:
        human_model = MovementModel()
        human_model.load_model(root_path)
        human_model.set_default_param()

        planner = MDPFixedTimePolicy(tmodel=human_model)

        planner.compute_policy(s_g, modality, max_iter=20)
    else:
        with open(root_path + "/mdp_planner.pkl") as f:
            planner = pickle.load(f)

    fig, axes = plt.subplots()
    planner.visualize_policy(axes)
    plt.show()

    if flag_plan:
        with open(root_path + "/mdp_planner.pkl", "w") as f:
            pickle.dump(planner, f)

    sim = Simulator(planner)
    n_trials = 30

    traj_list = []
    for i in range(n_trials):
        traj_list.append(sim.run_trial((0.0, 0.0, 0.0), s_g, modality, 15.0, tol=0.5))

    fig, axes = plt.subplots()
    for i in range(n_trials):
        t, traj = traj_list[i]
        axes.plot(traj[:, 0], traj[:, 1])
    axes.axis("equal")

    axes.scatter(s_g[0], s_g[1])

    plt.show()


if __name__ == "__main__":
    # simulate_naive_policy(30, np.array([3.0, 2.0, 0.0]), "haptic")
    validate_MDP_policy("/home/yuhang/Documents/proactive_guidance/training_data/user0", True)