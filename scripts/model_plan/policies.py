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
        with open(file_path + "/planner_env.pkl") as f:
            self.obs = pickle.load(f)

    def gen_env(self, obs_list, file_path=None):
        # obs_list is a list of rectangular obstacles in the form (x, y, w, h)
        for x, y, w, h in obs_list:
            for i in range(int(w / self.dx)):
                for j in range(int(h / self.dy)):
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

        # self.visualize_policy()
        # plt.show()

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
                    if xg == xgg and yg == ygg:
                        continue

                    x, y, alp = self.grid_to_xy(xg, yg, alp_g)

                    # iterate over all actions
                    Q_max = -1000.0
                    a_opt = 0.0
                    n_comm_opt = 0.0

                    # only sample actions that make sense
                    a_list = wrap_to_pi(da * np.arange(0, self.nA) - a_range / 2.0 + self.policy[xg, yg, alp_g])

                    for ai, a in enumerate(a_list):
                        Vnext = 0.0
                        n_comm_next = 0.0
                        for k in range(self.n_samples):
                            # sample a new state
                            self.tmodel.set_state(x, y, alp)
                            s_next = self.tmodel.sample_state((modality, a), 0.5, self.dt)
                            if s_next[0] < self.x_range[0] or s_next[0] >= self.x_range[1] or \
                                            s_next[1] < self.y_range[0] or s_next[1] >= self.y_range[1]:
                                Vnext += self.r_obs
                                n_comm_next += 20
                            else:
                                Vnext += self.get_value(self.V, s_next[0], s_next[1], s_next[2])
                                n_comm_next += self.get_value(self.n_comm, s_next[0], s_next[1], s_next[2])

                        # if Vnext != 0:
                        #     print "here"

                        self.Q[xg, yg, alp_g, ai] = self.gamma * Vnext / self.n_samples
                        n_comm_next /= self.n_samples

                        if self.Q[xg, yg, alp_g, ai] > Q_max:
                            Q_max = self.Q[xg, yg, alp_g, ai]
                            n_comm_opt = n_comm_next
                            a_opt = a

                    # update value function and policy
                    # only update value for non-obstacle states
                    if not self.obs[xg, yg]:
                        self.V[xg, yg, alp_g] = Q_max
                    self.n_comm[xg, yg, alp_g] = n_comm_opt + 1

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
        return self.get_value_func(s) < -20.0

    def visualize_policy(self, ax=None):
        Vplot = np.max(self.V, axis=2)
        for xg in range(self.nX):
            for yg in range(self.nY):
                if self.obs[xg, yg]:
                    Vplot[xg, yg] = -2

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


def validate_free_space_policy(planner, s_g, modality, path):
    fig, axes = plt.subplots()
    planner.visualize_policy(axes)
    fig.savefig(path + "/value_func.png")

    sim = Simulator(planner)
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


def generate_mdp_policies(protocol_file, model_path, modality):
    protocol_data = np.loadtxt(protocol_file, delimiter=", ")

    n_targets = int(np.max(protocol_data[:, 0], ) + 1)
    generated = np.zeros((n_targets, ))
    save_path = model_path + "/mdp_" + modality + "/free_space"

    ranges = [[-1.5, 4.0], [-1.0, 5.0]]

    for trial_data in protocol_data:
        target_id = int(trial_data[0])
        if generated[target_id] < 1.0:
            # load human model first
            with open(model_path + "/human_models/user0_default.pkl") as f:
                human_model = pickle.load(f)

            # create the planner
            mdp_policy = MDPFixedTimePolicy(human_model, ranges)

            # compute policy
            s_g = np.array([trial_data[2], trial_data[3], 0.0])
            mdp_policy.compute_policy(s_g, modality, max_iter=20)

            with open(save_path + "/target" + str(target_id) + ".pkl", "w") as f:
                pickle.dump(mdp_policy, f)

            # save some figures for debug
            fig_path = save_path + "/target" + str(target_id) + "_figs"
            mkdir_p(fig_path)
            validate_free_space_policy(mdp_policy, s_g, modality, fig_path)

            generated[target_id] = 1.0


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
    # simulate_naive_policy(30, np.array([3.0, 2.0, 0.0]), "haptic")
    # validate_MDP_policy("/home/yuhang/Documents/proactive_guidance/training_data/user0",
    #                     flag_with_obs=True, flag_plan=True)

    # generate_naive_policies("../../resources/protocols/free_space_exp_protocol_7targets.txt",
    #                         "../../resources/pretrained_models",
    #                         "haptic")
    #
    # generate_naive_policies("../../resources/protocols/free_space_exp_protocol_7targets.txt",
    #                         "../../resources/pretrained_models",
    #                         "audio")

    generate_mdp_policies("../../resources/protocols/free_space_exp_protocol_7targets.txt",
                          "../../resources/pretrained_models",
                          "haptic")

    generate_mdp_policies("../../resources/protocols/free_space_exp_protocol_7targets.txt",
                          "../../resources/pretrained_models",
                          "audio")
