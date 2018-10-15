#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import time
import pickle
from collections import deque

from pre_processing import wrap_to_pi
from movement_model import MovementModel, MovementModelParam
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
    def __init__(self, tmodel, ranges=None):
        # workspace size, resolution, offset
        self.nAlp = 12

        self.dx = 0.25
        self.dy = 0.25
        self.dalp = 2.0 * np.pi / self.nAlp

        self.alp_offset = -np.pi

        if ranges is None:
            self.nX = 20
            self.nY = 20

            self.x_offset = 0.0
            self.y_offset = 0.0

            self.x_range = [self.x_offset, self.x_offset + self.dx * (self.nX-1)]
            self.y_range = [self.y_offset, self.y_offset + self.dy * (self.nY-1)]
        else:
            self.x_range, self.y_range = ranges
            self.x_offset = self.x_range[0]
            self.y_offset = self.y_range[0]

            self.nX = int((self.x_range[1] - self.x_range[0]) / self.dx) + 1
            self.nY = int((self.y_range[1] - self.y_range[0]) / self.dy) + 1

        # fixed time interval
        self.dt = 2.0

        # number of actions
        self.nA = 11
        self.a_list = np.arange(0, self.nA) * (np.pi / self.nA) - np.pi

        self.a_range_init = 2.0 * np.pi
        self.a_range_final = np.pi / 3.0
        self.a_dec_iter = 10

        # value function, Q function, policy, rewards
        self.V = -1000.0 * np.ones((self.nX, self.nY, self.nAlp))
        self.n_comm = np.zeros((self.nX, self.nY, self.nAlp))
        self.Q = np.zeros((self.nX, self.nY, self.nAlp, self.nA))
        self.policy = np.zeros((self.nX, self.nY, self.nAlp))

        self.obs = np.zeros((self.nX, self.nY))

        self.r_goal = 10.0
        self.r_obs = -200.0

        # discount factor
        self.gamma = 0.9

        # policy update counting threshold
        self.n_update_th = 3

        # how many samples to draw for each update
        self.n_samples = 20

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
        if af >= self.nA:
            af -= self.nA

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
        with open(file_path + "/planner_env.pkl") as f:
            self.obs = pickle.load(f)

    def gen_env(self, obs_list, file_path=None):
        # obs_list is a list of rectangular obstacles in the form (x, y, w, h)
        for x, y, w, h in obs_list:
            for i in range(int(w / self.dx) + 1):
                for j in range(int(h / self.dy) + 1):
                    xi = x + i * self.dx
                    yj = y + j * self.dy

                    xg, yg, tmp = self.xy_to_grid(xi, yj, 0.0)
                    self.obs[xg, yg] = 1.0

        if file_path is not None:
            with open(file_path + "/planner_env.pkl", "w") as f:
                pickle.dump(self.obs, f)

    def init_value_function(self, s_g):
        # set obstacle values
        for xg in range(self.nX):
            for yg in range(self.nY):
                for alp_g in range(self.nAlp):
                    self.V[xg, yg, alp_g] = self.r_obs * self.obs[xg, yg]

        # set goal value
        xgg, ygg, tmp = self.xy_to_grid(s_g[0], s_g[1], 0.0)
        goal_states = [(xgg, ygg), (xgg+1, ygg), (xgg+1, ygg+1), (xgg, ygg+1)]

        # perform a BFS to initialize values of all states and obtain an update sequence
        visited = np.zeros((self.nX, self.nY))
        nodes = deque()

        for state in goal_states:
            self.V[state[0], state[1]] += self.r_goal
            visited[state[0], state[1]] = 1.0
            nodes.append((state[0], state[1]))

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
                        a = self.gamma * self.V[xg, yg, 0]
                        self.V[xg_next, yg_next, alp_g] = self.gamma * self.V[xg, yg, 0]

                        # apply naive policy
                        x, y, alp = self.grid_to_xy(xg_next, yg_next, alp_g)
                        self.policy[xg_next, yg_next, alp_g] = wrap_to_pi(np.arctan2(-dy[i], -dx[i]) - alp)

        # do a second pass through to handle obstacles
        nodes = deque()
        for xg in range(self.nX):
            for yg in range(self.nY):
                if visited[xg, yg]:
                    nodes.append((xg, yg))

        while nodes:
            xg, yg = nodes.popleft()

            for i in range(8):
                xg_next = xg + dx[i]
                yg_next = yg + dy[i]

                if xg_next < 0 or yg_next < 0 or xg_next >= self.nX or yg_next >= self.nY:
                    continue

                if not visited[xg_next, yg_next]:
                    visited[xg_next, yg_next] = True
                    nodes.append((xg_next, yg_next))
                    self.update_q.append((xg_next, yg_next))

                    # update value and initial policy
                    for alp_g in range(self.nAlp):
                        # apply naive policy
                        x, y, alp = self.grid_to_xy(xg_next, yg_next, alp_g)
                        self.policy[xg_next, yg_next, alp_g] = wrap_to_pi(np.arctan2(-dy[i], -dx[i]) - alp)

        # self.visualize_policy()
        # plt.show()

        # initialize the policies at goal states
        for i, state in enumerate(goal_states):
            a_opt = np.pi / 4.0 + i * 0.5 * np.pi
            for ag in range(self.nAlp):
                alp = self.alp_offset + ag * self.dalp
                self.policy[state[0], state[1], ag] = wrap_to_pi(a_opt - alp)

    def compute_policy(self, s_g, modality, max_iter=50):
        if self.flag_policy_computed and self.modality == modality:
            goal_diff = np.linalg.norm(self.s_g[:2] - s_g[:2])
            if goal_diff < 0.25:
                print "No need to recompute policy"
                return

        # adjust s_g to be the center of the grid
        xgg, ygg, tmp = self.xy_to_grid(s_g[0], s_g[1], 0.0)
        s_g_min = self.grid_to_xy(xgg, ygg, tmp)
        s_g_max = self.grid_to_xy(xgg+1, ygg+1, tmp)

        self.s_g = 0.5 * (np.asarray(s_g_min) + np.asarray(s_g_max))
        print self.s_g

        # self.s_g = s_g
        self.modality = modality

        if modality == "haptic":
            self.dt = 2.0
        else:
            self.dt = 3.0

        self.init_value_function(self.s_g)

        counter_policy_not_updated = 0
        for i in range(max_iter):
            print "At iteration ", i, "..."

            flag_policy_update = False
            # V_curr = self.V
            # self.V = np.zeros((self.nX, self.nY, self.nAlp))

            if i <= self.a_dec_iter:
                a_range = self.a_range_init - (self.a_range_init - self.a_range_final) / self.a_dec_iter * i
            else:
                a_range = self.a_range_final

            da = a_range / self.nA
            print da

            # iterate over all states
            for xg, yg in self.update_q:
                for alp_g in range(self.nAlp):
                    # don't perform update on goal state
                    # if self.obs[xg, yg] > 0:
                    #     continue
                    if (xg == xgg or xg == xgg+1) and (yg == ygg or yg == ygg+1):
                        continue

                    x, y, alp = self.grid_to_xy(xg, yg, alp_g)
                    d = np.linalg.norm(np.array([x, y]) - self.s_g[:2])

                    # iterate over all actions
                    Q_max = -1000.0
                    a_opt = 0.0
                    n_comm_opt = 0.0

                    # only sample actions that make sense
                    a_list = wrap_to_pi(da * np.arange(0, self.nA) - a_range / 2.0 + self.policy[xg, yg, alp_g])

                    for ai, a in enumerate(a_list):
                        Vnext = 0.0
                        n_comm_next = 0.0

                        if i < max_iter-0:
                            n_samples = self.n_samples
                        else:
                            n_samples = 2 * self.n_samples

                        for k in range(n_samples):
                            # sample a new state
                            self.tmodel.set_state(x, y, alp)

                            if d < 0.5:
                                s_next = self.tmodel.sample_state((modality, a), 0.5, self.dt,
                                                                  flag_check_stop=True, s_g=self.s_g)
                                # if s_next[1] == 2.25:
                                #     print s_next
                            else:
                                s_next = self.tmodel.sample_state((modality, a), 0.5, self.dt)

                            if s_next[0] < self.x_range[0] or s_next[0] >= self.x_range[1] or \
                                            s_next[1] < self.y_range[0] or s_next[1] >= self.y_range[1]:
                                Vnext += self.r_obs
                                n_comm_next += 20
                            else:
                                Vnext += self.get_value(self.V, s_next[0], s_next[1], s_next[2])
                                n_comm_next += self.get_value(self.n_comm, s_next[0], s_next[1], s_next[2])

                                if self.is_obs(s_next):
                                    Vnext += self.r_obs

                        # if Vnext != 0:
                        #     print "here"

                        self.Q[xg, yg, alp_g, ai] = self.gamma * Vnext / n_samples
                        n_comm_next /= n_samples

                        if self.Q[xg, yg, alp_g, ai] > Q_max:
                            Q_max = self.Q[xg, yg, alp_g, ai]
                            n_comm_opt = n_comm_next
                            a_opt = a

                    # update value function and policy
                    # only update value for non-obstacle states
                    self.V[xg, yg, alp_g] = Q_max
                    self.n_comm[xg, yg, alp_g] = n_comm_opt + 1

                    if self.obs[xg, yg] > 0:
                        self.V[xg, yg, alp_g] += self.r_obs

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

    def get_value_func(self, s):
        return self.get_value(self.V, s[0], s[1], wrap_to_pi(s[2]))

    def get_n_comm(self, s):
        return self.get_value(self.n_comm, s[0], s[1], wrap_to_pi(s[2]))

    def is_obs(self, s):
        xg, yg, tmp = self.xy_to_grid(s[0], s[1], s[2])
        return self.obs[xg, yg] > 0.0
        # return self.get_value_func(s) < -20.0

    def visualize_policy(self, ax=None):
        Vplot = np.max(self.V, axis=2)
        for xg in range(self.nX):
            for yg in range(self.nY):
                if self.obs[xg, yg]:
                    Vplot[xg, yg] = -2

        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(Vplot.transpose(), origin="lower", cmap="hot", interpolation="nearest")
        
        
class NaivePolicyObs(MDPFixedTimePolicy):
    def __init__(self, ranges):
        super(NaivePolicyObs, self).__init__(None, ranges)
        self.dinc = 0.25
        self.gamma = 0.95

    def compute_policy(self, s_g, modality, max_iter=50):
        if self.flag_policy_computed and self.modality == modality:
            goal_diff = np.linalg.norm(self.s_g[:2] - s_g[:2])
            if goal_diff < 0.25:
                print "No need to recompute policy"
                return

        # adjust s_g to be the center of the grid
        xgg, ygg, tmp = self.xy_to_grid(s_g[0], s_g[1], 0.0)
        s_g_min = self.grid_to_xy(xgg, ygg, tmp)
        s_g_max = self.grid_to_xy(xgg+1, ygg+1, tmp)

        self.s_g = 0.5 * (np.asarray(s_g_min) + np.asarray(s_g_max))
        print self.s_g

        # self.s_g = s_g
        self.modality = modality

        self.init_value_function(self.s_g)

        counter_policy_not_updated = 0
        for i in range(max_iter):
            print "At iteration ", i, "..."

            flag_policy_update = False

            if i <= self.a_dec_iter:
                a_range = self.a_range_init - (self.a_range_init - self.a_range_final) / self.a_dec_iter * i
            else:
                a_range = self.a_range_final

            da = a_range / self.nA
            print da

            # iterate over all states
            for xg, yg in self.update_q:
                for alp_g in range(self.nAlp):
                    # don't perform update on goal state
                    if (xg == xgg or xg == xgg+1) and (yg == ygg or yg == ygg+1):
                        continue

                    x, y, alp = self.grid_to_xy(xg, yg, alp_g)

                    # iterate over all actions
                    Q_max = -1000.0
                    a_opt = 0.0

                    # only sample actions that make sense
                    a_list = wrap_to_pi(da * np.arange(0, self.nA) - a_range / 2.0 + self.policy[xg, yg, alp_g])

                    for ai, a in enumerate(a_list):
                        s_next = np.zeros((3, ))
                        s_next[0] = x + self.dinc * np.cos(a+alp)
                        s_next[1] = y + self.dinc * np.sin(a+alp)
                        s_next[2] = a+alp

                        if s_next[0] < self.x_range[0] or s_next[0] >= self.x_range[1] or \
                                        s_next[1] < self.y_range[0] or s_next[1] >= self.y_range[1]:
                            self.Q[xg, yg, alp_g, ai] = self.r_obs
                        else:
                            self.Q[xg, yg, alp_g, ai] = self.gamma * \
                                                        self.get_value(self.V, s_next[0], s_next[1], s_next[2])

                            if self.is_obs(s_next):
                                # punish if end up in an obstacle state
                                self.Q[xg, yg, alp_g, ai] += self.r_obs

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


def simulate_naive_policy(n_trials, s_g, modality, usr):
    planner = NaivePolicy()

    model_path = "/home/yuhang/Documents/proactive_guidance/training_data/user" + str(usr)
    sim = Simulator(planner, model_path)

    traj_list = []
    for i in range(n_trials):
        traj_list.append(sim.run_trial((-1.0, 2.0, 0.0), s_g, modality, 20.0, tol=0.5))

    fig, axes = plt.subplots()
    for i in range(n_trials):
        t, traj = traj_list[i]
        axes.plot(traj[:, 0], traj[:, 1])
    axes.axis("equal")

    axes.scatter(s_g[0], s_g[1])

    plt.show()


def validate_MDP_policy(root_path, flag_with_obs=True, flag_plan=True):
    s_g = np.array([4.0, 3.0, 0.0])
    modality = "haptic"

    obs_list = [(1.5, 2.00, 1.0, 1.25),
                (2.0, 1.0, 1.25, 0.5)]

    if flag_with_obs:
        file_name = root_path + "/mdp_planenr_obs_" + modality + ".pkl"
    else:
        file_name = root_path + "/mdp_planner_" + modality + ".pkl"

    if flag_plan:
        human_model = MovementModel()
        human_model.load_model(root_path)
        human_model.set_default_param()

        planner = MDPFixedTimePolicy(tmodel=human_model)

        if flag_with_obs:
            planner.gen_env(obs_list)

        planner.compute_policy(s_g, modality, max_iter=30)
    else:
        with open(file_name) as f:
            planner = pickle.load(f)

    fig, axes = plt.subplots()
    planner.visualize_policy(axes)
    plt.show()

    if flag_plan:
        with open(file_name, "w") as f:
            pickle.dump(planner, f)

    sim = Simulator(planner)
    n_trials = 30

    traj_list = []
    start_time = time.time()
    for i in range(n_trials):
        traj_list.append(sim.run_trial((0.5, 0.5, 0.0), s_g, modality, 30.0, tol=0.5))

    print "--- %s seconds ---" % (time.time() - start_time)

    fig, axes = plt.subplots()
    for i in range(n_trials):
        t, traj = traj_list[i]
        axes.plot(traj[:, 0], traj[:, 1])
    axes.axis("equal")

    axes.scatter(s_g[0], s_g[1])

    if flag_with_obs:
        for x, y, w, h in obs_list:
            rect = Rectangle((x, y), w, h)
            axes.add_patch(rect)

    plt.show()


def validate_free_space_policy(planner, s_g, modality, path, model_path):
    fig, axes = plt.subplots()
    planner.visualize_policy(axes)
    fig.savefig(path + "/value_func.png")

    sim = Simulator(planner, model_path)
    n_trials = 30

    traj_list = []
    for i in range(n_trials):
        traj_list.append(sim.run_trial((-1.0, 2.0, 0.0), s_g, modality, 30.0, tol=0.5))

    fig, axes = plt.subplots()
    for i in range(n_trials):
        t, traj = traj_list[i]
        axes.plot(traj[:, 0], traj[:, 1])
    axes.axis("equal")

    axes.scatter(s_g[0], s_g[1])
    fig.savefig(path + "/simulation.png")


def validate_obs_policy(planner, s_g, env_list, modality, path, model_path):
    fig, axes = plt.subplots()
    planner.visualize_policy(axes)
    fig.savefig(path + "/value_func.png")

    sim = Simulator(planner, model_path)
    n_trials = 30

    traj_list = []
    for i in range(n_trials):
        traj_list.append(sim.run_trial((-2.0, 0.5, 0.25 * np.pi), s_g, modality, 30.0, tol=0.5))

    fig, axes = plt.subplots()
    for i in range(n_trials):
        t, traj = traj_list[i]
        axes.plot(traj[:, 0], traj[:, 1])

    target, obs_list = env_list

    for x, y, w, h in obs_list:
        rect = Rectangle((x, y), w, h)
        axes.add_patch(rect)

    axes.axis("equal")

    # axes.scatter(s_g[0], s_g[1], facecolor='r')
    rect = Rectangle((s_g[0], s_g[1]), 0.25, 0.25, facecolor='r', lw=0)
    axes.add_patch(rect)
    fig.savefig(path + "/simulation.png")


def generate_naive_policies(protocol_file, save_path, modality):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_targets = int(np.max(protocol_data[:, 0], ) + 1)
    generated = np.zeros((n_targets, ))

    save_path += "/naive_" + modality + "/free_space"
    for trial_data in protocol_data:
        target_id = int(trial_data[0])
        if generated[target_id] < 1.0:
            naive_policy = NaivePolicy()
            naive_policy.compute_policy(np.array([trial_data[2], trial_data[3], 0.0]), modality)

            with open(save_path + "/target" + str(target_id) + ".pkl", "w") as f:
                pickle.dump(naive_policy, f)

            generated[target_id] = 1.0


def generate_naive_policies_from_mdp(policy_path, modality, n_target=7):
    for target in range(n_target):
        with open(policy_path + "/mdp_" + modality + "/free_space/target" + str(target) + ".pkl") as f:
            mdp_policy = pickle.load(f)

        naive_policy = NaivePolicy()
        naive_policy.compute_policy(mdp_policy.s_g, modality)

        with open(policy_path + "/naive_" + modality + "/free_space/target" + str(target) + ".pkl", "w") as f:
            pickle.dump(naive_policy, f)


def generate_mdp_policies(protocol_file, model_path, modality, usr):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_targets = int(np.max(protocol_data[:, 0], ) + 1)
    generated = np.zeros((n_targets, ))
    save_path = model_path + "/pretrained_model/mdp_" + modality + "/free_space"

    ranges = [[-1.5, 4.5], [-1.5, 5.5]]

    for trial_data in protocol_data:
        target_id = int(trial_data[0])
        if generated[target_id] < 1.0:
            # load human model first
            with open("../resources/pretrained_models/human_models/user" + str(usr) + "_default.pkl") as f:
                human_model = pickle.load(f)

            # create the planner
            mdp_policy = MDPFixedTimePolicy(human_model, ranges)

            # compute policy
            s_g = np.array([trial_data[2], trial_data[3], 0.0])
            mdp_policy.compute_policy(s_g, modality, max_iter=20)

            mkdir_p(save_path)
            with open(save_path + "/target" + str(target_id) + ".pkl", "w") as f:
                pickle.dump(mdp_policy, f)

            # save some figures for debug
            fig_path = save_path + "/target" + str(target_id) + "_figs"
            mkdir_p(fig_path)
            validate_free_space_policy(mdp_policy, s_g, modality, fig_path,
                                       model_path)

            generated[target_id] = 1.0


def generate_policies_with_obs(env_list, model_path, modality, usr, policy="mdp"):
    n_targets = len(env_list)

    save_path = model_path + "/user" + str(usr) + "/pretrained_model/" + policy + "_" + modality + "/obstacle"

    ranges = [[-3.0, 4.5], [-1.0, 6.0]]

    for i in range(0, n_targets):
        target_pos, obs_list = env_list[i]

        # load human model first
        with open("../resources/pretrained_models/human_models/user" + str(usr) + "_default.pkl") as f:
            human_model = pickle.load(f)

        # create the planner
        if policy == "mdp":
            policy_planner = MDPFixedTimePolicy(human_model, ranges)
        else:
            policy_planner = NaivePolicyObs(ranges)

        policy_planner.gen_env(obs_list)

        # compute policy
        s_g = np.array([target_pos[0], target_pos[1], 0.0])
        policy_planner.compute_policy(s_g, modality, max_iter=20)

        mkdir_p(save_path)
        with open(save_path + "/target" + str(i) + ".pkl", "w") as f:
            pickle.dump(policy_planner, f)

        # save some figures for debug
        fig_path = save_path + "/target" + str(i) + "_figs"
        mkdir_p(fig_path)
        validate_obs_policy(policy_planner, s_g, env_list[i], modality, fig_path,
                            model_path + "/user" + str(usr))


def mkdir_p(mypath):
    """
    Creates a directory. equivalent to using mkdir -p on the command line
    """

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:
        # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


if __name__ == "__main__":
    simulate_naive_policy(30, np.array([2.46, 4.00, 0.0]), "haptic", 1)
    # validate_MDP_policy("/home/yuhang/Documents/proactive_guidance/training_data/user0",
    #                     flag_with_obs=True, flag_plan=True)

    # generate_naive_policies("../../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                         "../../resources/pretrained_models",
    #                         "haptic")
    #
    # generate_naive_policies("../../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                         "../../resources/pretrained_models",
    #                         "audio")

    # generate_mdp_policies("../../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                       "../../resources/pretrained_models",
    #                       "haptic")
    #
    # generate_mdp_policies("../../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                       "../../resources/pretrained_models",
    #                       "audio")

    # generate_mdp_policies("../../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                       "/home/yuhang/Documents/proactive_guidance/training_data/user1/pretrained_model",
    #                       "haptic", 1)
