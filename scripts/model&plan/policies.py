#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from pre_processing import wrap_to_pi


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
        self.nX = 15
        self.nY = 15

        self.dx = 0.5
        self.dy = 0.5

        self.x_offset = 0.0
        self.y_offset = 0.0

        self.x_range = [self.x_offset, self.x_offset + self.dx * self.nX]
        self.y_range = [self.y_offset, self.y_offset + self.dy * self.nY]

        # fixed time interval
        self.dt = 2.0

        # number of actions
        self.nA = 12
        self.a_lsit = np.arange(0, self.nA) * (np.pi / self.nA) - np.pi

        # value function, Q function, policy, rewards
        self.V = -1000.0 * np.ones((self.nX, self.nY))
        self.Q = np.zeros((self.nX, self.nY, self.nA))
        self.policy = np.zeros((self.nX, self.nY))

        self.obs = np.zeros((self.nX, self.nY))

        self.r_goal = 10.0
        self.r_obs = -20.0

        # discount factor
        self.gamma = 0.9

        # policy update counting threshold
        self.n_update_th = 10

        # how many samples to draw for each update
        self.n_samples = 10

        # transition model for sampling
        self.tmodel = tmodel

        # to prevent repeatedly compute policy for same goal
        self.s_g = None
        self.modality = None
        self.flag_policy_computed = False

    # helper functions
    def xy_to_grid(self, x, y):
        xg = int((x - self.x_offset) / self.dx)
        yg = int((y - self.y_offset) / self.dy)

        return xg, yg

    def grid_to_xy(self, xg, yg):
        x = xg * self.dx + self.x_offset
        y = yg * self.dy + self.y_offset

        return x, y

    def get_value(self, f, x, y):
        xf = (x - self.x_offset) / self.dx
        yf = (y - self.y_offset) / self.dy
        xg = int(xf)
        yg = int(yf)

        xf -= xg
        yf -= yg

        f1 = (1.0 - xf) * f[xg, yg] + xf * f[xg+1, yg]
        f2 = (1.0 - xf) * f[xg, yg+1] + xf * f[xg+1, yg+1]

        return (1.0 - yf) * f1 + yf * f2

    def load_env(self, file_path):
        # load possible obstacles
        pass

    def init_value_function(self, s_g):
        # set obstacle values
        self.V = self.r_obs * self.obs

        # set goal value
        xgg, ygg = self.xy_to_grid(s_g[0], s_g[1])
        self.V[xgg, ygg] = self.r_goal

    def compute_policy(self, s_g, modality, max_iter=100):
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
        xgg, ygg = self.xy_to_grid(s_g[0], s_g[1])

        counter_policy_not_updated = 0
        for i in range(max_iter):
            print "At iteration ", i, "..."

            flag_policy_update = False
            V_curr = self.V
            self.V = np.zeros((self.nX, self.nY))

            # iterate over all states
            for xg in range(self.nX):
                for yg in range(self.nY):
                    # don't perform update on obstacle and goal state
                    if self.obs[xg][yg] > 0:
                        continue
                    if xg == xgg and yg == ygg:
                        continue

                    x, y = self.grid_to_xy(xg, yg)

                    # iterate over all actions
                    Q_max = -1000.0
                    a_opt = 0.0
                    for ai, a in enumerate(self.a_lsit):
                        Vnext = 0.0
                        for k in range(self.n_samples):
                            # sample a new state
                            # self.tmodel.set_state(x, y, 0.0)
                            s_next = self.tmodel.sample_state((modality, a), 0.5, self.dt)
                            if s_next[0] < self.x_range[0] or s_next[0] > self.x_range[1] or \
                                            s_next[1] < self.y_range[0] or s_next[1] > self.y_range[1]:
                                continue

                            Vnext += self.get_value(V_curr, s_next[0], s_next[1])

                        if Vnext != 0:
                            print "here"

                        self.Q[xg, yg, ai] = self.gamma * Vnext / self.n_samples

                        if self.Q[xg, yg, ai] > Q_max:
                            Q_max = self.Q[xg, yg, ai]
                            a_opt = a

                    # update value function and policy
                    self.V[xg, yg] = Q_max
                    if self.policy[xg, yg] != a_opt:
                        self.policy[xg, yg] = a_opt
                        flag_policy_update = True

            if not flag_policy_update:
                counter_policy_not_updated += 1
                if counter_policy_not_updated >= self.n_update_th:
                    print "Policy converged!"
                    break
            else:
                counter_policy_not_updated = 0

        self.flag_policy_computed = True

    def sample_policy(self, s):
        return self.get_value(self.policy, s[0], s[1])


if __name__ == "__main__":
    pass
