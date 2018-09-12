#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import pickle
import time

from pre_processing import wrap_to_pi
from movement_model import MovementModel, MovementModelParam
from gp_model_approx import GPModelApproxBase
from simulation import Simulator
from policies import MDPFixedTimePolicy


class MCTSNodeBase(object):
    def __init__(self, node_type):
        self.type = node_type
        self.count = 0
        self.depth = 0

        self.children = []
        self.parent = None

        self.value = 0.0
        self.n_comm = 0
        self.score = 0.0


class MCTSStateNode(MCTSNodeBase):
    def __init__(self, s, belief, t):
        super(MCTSStateNode, self).__init__("state")

        self.s = s
        self.t = t
        self.alp_d_mean, self.alp_d_std = belief

        self.a_list = []
        self.n_a = 0

        self.goal_reached = False


class MCTSActionNode(MCTSNodeBase):
    def __init__(self, s, a, t):
        super(MCTSActionNode, self).__init__("action")

        self.a = a
        self.s = s
        self.t = t


class MCTSTrajNode(MCTSNodeBase):
    def __init__(self, alp_d, traj):
        super(MCTSTrajNode, self).__init__("traj")

        self.alp_d = alp_d
        self.traj = traj
        self.selected = np.zeros_like(traj[0], dtype=int)

        self.goal_reached = False


class MCTSPolicy(object):
    def __init__(self, model, default_policy, modality):
        self.policy_tree_root = None
        self.root_no_comm = None

        self.model = model
        self.default_policy = default_policy
        self.modality = modality

        # goal, range
        self.s_g = self.default_policy.s_g
        self.x_range = self.default_policy.x_range
        self.y_range = self.default_policy.y_range

        self.goal_reaching_th = 0.5

        # obstacle
        # TODO: how to handle?

        # other settings
        self.n_actions_per_state = 5
        self.da_sample = np.deg2rad(10)

        self.action_widen_factor = 0.2
        self.traj_widen_factor = 0.2
        self.time_widen_factor = 0.2
        self.traj_widen_factor_no_comm = 0.2

        self.uct_weight = 1

        self.traj_sample_dt = 0.5
        self.traj_sample_T = 5.0

        self.time_sample_factor1 = 5.0
        self.time_sample_factor2 = -2.0

        self.gamma = 0.95
        self.gamma_scale = 0.5

        self.score_w_value = 1.0
        self.score_w_comm = -0.5

    def gen_action_list(self, s):
        # want to uniformly explore actions
        a_default = self.default_policy.sample_policy(s)
        a_list = [a_default]

        for i in range((self.n_actions_per_state - 1) / 2):
            a_list.append(a_default - (i+1) * self.da_sample)
            a_list.append(a_default + (i+1) * self.da_sample)

        return a_list

    def select_by_uct(self, node):
        uct = []

        for child in node.children:
            uct_score = child.score / child.count + self.uct_weight * np.sqrt(node.count / (child.count + 1))
            uct.append(uct_score)

        return np.argmax(uct)

    def select_action(self, state_node):
        state_node.count += 1

        # generate sampling list if none
        if not state_node.a_list:
            state_node.a_list = self.gen_action_list(state_node.s)

        # decide to sample a new action or sample from existing ones
        if state_node.n_a < len(state_node.a_list) and\
                        np.power(state_node.count, self.action_widen_factor) > state_node.n_a:
            a = state_node.a_list[state_node.n_a]
            state_node.n_a += 1

            action_node = MCTSActionNode(state_node.s, a, state_node.t)
            action_node.parent = state_node
            state_node.children.append(action_node)
        else:
            # compute uct scores for all actions
            child_id = self.select_by_uct(state_node)
            action_node = state_node.children[child_id]

        return action_node

    def check_traj(self, t, traj):
        T = len(t)
        for i in range(T):
            # check goal
            err = np.linalg.norm(self.s_g[:2] - traj[i, :2])
            if err < self.goal_reaching_th:
                return i+1, True

            # check oob
            if traj[i, 0] < self.x_range[0] or traj[i, 0] >= self.x_range[1] \
                    or traj[i, 1] < self.y_range[0] or traj[i, 1] >= self.y_range[1]:
                return i, False

        return T, False

    def select_traj(self, action_node):
        action_node.count += 1

        # decide to sample a new trajectory or use existing ones
        n = len(action_node.children)
        if np.power(action_node.count, self.traj_widen_factor) > n:
            self.model.set_state(action_node.s[0], action_node.s[1], action_node.s[2])
            alp_d, t, traj = self.model.sample_traj_single_action((self.modality, action_node.a),
                                                                  self.traj_sample_dt,
                                                                  self.traj_sample_T,
                                                                  flag_return_alp=True)
            t += action_node.t

            # check trajectory
            tf, flag_goal_reached = self.check_traj(t, traj)

            if flag_goal_reached:
                traj_node = MCTSTrajNode(alp_d, (t[2:tf], traj[2:tf]))
                traj_node.goal_reached = True
                traj_node.parent = action_node
                action_node.children.append(traj_node)
            else:
                # oob immediately
                if tf <= 3:
                    return None

                traj_node = MCTSTrajNode(alp_d, (t[2:tf], traj[2:tf]))
                traj_node.parent = action_node
                action_node.children.append(traj_node)
        else:
            chances = []
            for child in action_node.children:
                chances.append(self.model.get_prob_alp_d((self.modality, action_node.a), child.alp_d))
            chances = np.asarray(chances) / np.sum(chances)

            child_id = np.random.choice(np.arange(n), p=chances)
            traj_node = action_node.children[child_id]

        return traj_node

    def sample_time(self, traj_node, flag_new_node):
        elm = []
        chances = []

        v0 = self.default_policy.get_value_func(traj_node.traj[1][0])

        for i in range(1, len(traj_node.traj[0])):
            if (traj_node.selected[i] > 0) ^ flag_new_node:
                s = np.array([traj_node.traj[1][i, 0], traj_node.traj[1][i, 1], traj_node.alp_d])
                v = self.default_policy.get_value_func(s)
                alp_d = self.default_policy.sample_policy(s)
                alp_diff = wrap_to_pi(traj_node.alp_d - alp_d)

                t = traj_node.traj[0][i] - traj_node.traj[0][0]
                val = self.time_sample_factor1 * (v * self.gamma**(t * self.gamma_scale) - v0) +\
                      self.time_sample_factor2 * alp_diff**2

                elm.append(i)
                chances.append(np.exp(val))

        chances = np.asarray(chances) / np.sum(chances)

        if flag_new_node:
            if len(elm) == 0:
                print "this shouldn't happen"
            return np.random.choice(elm, p=chances)
        else:
            return np.random.choice(np.arange(len(traj_node.children)), p=chances)

    def select_time(self, traj_node):
        traj_node.count += 1

        n = len(traj_node.children)

        # specially handle goal reached condition
        if traj_node.goal_reached:
            if n > 0:
                return traj_node.children[0]
            else:
                state_node = MCTSStateNode(traj_node.traj[1][-1], (traj_node.alp_d, 0.0), traj_node.traj[0][-1])
                state_node.goal_reached = True
                state_node.parent = traj_node
                traj_node.children.append(state_node)
                return state_node
        else:
            if np.power(traj_node.count, self.time_widen_factor) > n:
                i = self.sample_time(traj_node, True)
                traj_node.selected[i] = 1
                state_node = MCTSStateNode(traj_node.traj[1][i], (traj_node.alp_d, 0.0), traj_node.traj[0][i])
                state_node.parent = traj_node
                traj_node.children.append(state_node)
            else:
                i = self.sample_time(traj_node, False)
                state_node = traj_node.children[i]

        return state_node

    def grow_tree(self):
        # sample an unvisited state
        node = self.policy_tree_root
        while node.count > 0 and not node.goal_reached:
            a_node = self.select_action(node)
            traj_node = self.select_traj(a_node)

            if traj_node is None:
                a_node.value -= 100.0
                a_node.score = self.score_w_value * a_node.value + self.score_w_comm * a_node.n_comm
                break

            node = self.select_time(traj_node)

        # instead of run real roll out, use value functions from offline policy
        t_curr = node.t
        v = self.default_policy.get_value_func(node.s)
        n_comm = self.default_policy.get_n_comm(node.s)

        if node.count == 0:
            node.count = 1

        while node.parent is not None:
            traj_node = node.parent
            # traj_node.count += 1

            a_node = traj_node.parent
            # a_node.count += 1
            a_node.value += self.gamma**((t_curr - a_node.t) * self.gamma_scale) * v
            a_node.n_comm += n_comm + 1
            n_comm += 1

            a_node.score = self.score_w_value * a_node.value + self.score_w_comm * a_node.n_comm

            node = a_node.parent
            # node.count += 1

    def select_traj_no_comm(self, state_node, action_node):
        state_node.count += 1
        action_node.count += 1

        # decide to sample a new trajectory or use existing ones
        n = len(action_node.children)
        if np.power(action_node.count, self.traj_widen_factor_no_comm) > n:
            # sample alp_d
            alp_d = np.random.normal(state_node.alp_d_mean, state_node.alp_d_std)

            self.model.set_state(action_node.s[0], action_node.s[1], action_node.s[2])
            t, traj = self.model.sample_traj_no_action((self.modality, alp_d),
                                                        self.traj_sample_dt,
                                                        self.traj_sample_T)
            t += action_node.t

            # check trajectory
            tf, flag_goal_reached = self.check_traj(t, traj)

            if flag_goal_reached:
                traj_node = MCTSTrajNode(alp_d, (t[:tf], traj[:tf]))
                traj_node.goal_reached = True
                traj_node.parent = action_node
                action_node.children.append(traj_node)
            else:
                traj_node = MCTSTrajNode(alp_d, (t[:tf], traj[:tf]))
                traj_node.parent = action_node
                action_node.children.append(traj_node)
        else:
            chances = []
            for child in action_node.children:
                # compute chance based on Gaussian pdf
                var = state_node.alp_d_std ** 2
                a_diff = child.alp_d - state_node.alp_d_mean

                chances.append(np.exp(-a_diff**2 / 2.0 / var) / np.sqrt(2.0 * np.pi * var))
            chances = np.asarray(chances) / np.sum(chances)

            child_id = np.random.choice(np.arange(n), p=chances)
            traj_node = action_node.children[child_id]

        return traj_node

    def grow_tree_no_comm(self):
        # sample an unvisited state
        node = self.root_no_comm
        action_node = self.root_no_comm.children[0]
        traj_node = self.select_traj_no_comm(node, action_node)
        node = self.select_time(traj_node)

        while node.count > 0 and not node.goal_reached:
            a_node = self.select_action(node)
            traj_node = self.select_traj(a_node)

            if traj_node is None:
                a_node.value -= 100.0
                a_node.score = self.score_w_value * a_node.value + self.score_w_comm * a_node.n_comm
                break

            node = self.select_time(traj_node)

        # instead of run real roll out, use value functions from offline policy
        t_curr = node.t
        v = self.default_policy.get_value_func(node.s)
        n_comm = self.default_policy.get_n_comm(node.s)

        if node.count == 0:
            node.count = 1

        while node.parent is not None:
            traj_node = node.parent
            # traj_node.count += 1

            a_node = traj_node.parent
            # a_node.count += 1
            a_node.value += self.gamma**((t_curr - a_node.t) * self.gamma_scale) * v

            if a_node.a is None:
                a_node.n_comm += n_comm
            else:
                a_node.n_comm += n_comm + 1
                n_comm += 1

            a_node.score = self.score_w_value * a_node.value + self.score_w_comm * a_node.n_comm

            node = a_node.parent
            # node.count += 1

    def generate_policy(self, s_init, t_max=0.8):
        start_time = time.time()

        # construct a root
        self.policy_tree_root = MCTSStateNode(s_init, (0.0, 0.0), 0.0)

        # keep growing tree until time reached
        t = time.time() - start_time
        counter = 0
        while t < t_max:
            self.grow_tree()
            counter += 1
            t = time.time() - start_time

        print "Has grown tree ", counter, " times"

    def generate_policy_no_comm(self, s_init, b_init, t_max=0.8):
        start_time = time.time()

        # create a root node and a fake action?
        self.root_no_comm = MCTSStateNode(s_init, b_init, 0.0)

        no_comm_action_node = MCTSActionNode(s_init, None, 0.0)
        no_comm_action_node.parent = self.root_no_comm
        self.root_no_comm.children.append(no_comm_action_node)

        t = time.time() - start_time
        counter = 0
        while t < t_max:
            self.grow_tree_no_comm()
            counter += 1
            t = time.time() - start_time

        print "Has grown tree ", counter, " times"

    def get_policy(self):
        a_opt = None
        v_max = -1000.0
        for child in self.policy_tree_root.children:
            if child.score / child.count > v_max:
                v_max = child.score / child.count
                a_opt = child.a

        return a_opt, v_max

    def get_policy_no_comm(self):
        v = self.root_no_comm.children[0].score / self.root_no_comm.children[0].count
        return v

    def visualize_search_tree(self, node, ax):
        if node.type == "state":
            # plot a point
            if node.goal_reached:
                ax.scatter(node.s[0], node.s[1], c='r')
            else:
                ax.scatter(node.s[0], node.s[1])
        elif node.type == "action":
            # plot nothing for now
            pass
        elif node.type == "traj":
            ax.plot(node.traj[1][:, 0], node.traj[1][:, 1])

        if node.children:
            for child in node.children:
                self.visualize_search_tree(child, ax)

    def visualize_best_path(self, node, ax):
        # plot a point
        if node.goal_reached:
            ax.scatter(node.s[0], node.s[1], c='r')
        else:
            ax.scatter(node.s[0], node.s[1])

        if not node.children:
            return

        action_node = None
        v_max = -1000.0
        for child in node.children:
            if child.score / child.count > v_max:
                v_max = child.score / child.count
                action_node = child
        print "action values are: ", v_max, action_node.value / action_node.count, action_node.n_comm / action_node.count

        chances = []
        for child in action_node.children:
            chances.append(child.count)

        i = np.argmax(chances)
        traj_node = action_node.children[i]
        ax.plot(traj_node.traj[1][:, 0], traj_node.traj[1][:, 1])

        chances = []
        for child in traj_node.children:
            chances.append(child.count)

        i = np.argmax(chances)
        node = traj_node.children[i]

        self.visualize_best_path(node, ax)


def policy_search_example(policy_path, modality, flag_no_comm=False):
    # create human model and default policy
    with open(policy_path) as f:
        mdp_policy = pickle.load(f)

    human_model = mdp_policy.tmodel

    # create a MCTS policy
    mcts_policy = MCTSPolicy(human_model, mdp_policy, modality)

    print "Goal is: ", mcts_policy.s_g

    # generate and visualize tree
    if flag_no_comm:
        s_init = np.array([0.5, 0.5, 0.0])
        b_init = (0.0, 0.1)
        mcts_policy.generate_policy_no_comm(s_init, b_init, t_max=0.8)

        fig, ax = plt.subplots()
        mcts_policy.visualize_search_tree(mcts_policy.root_no_comm, ax)
        ax.axis("equal")

        plt.show()

        fig, ax = plt.subplots()
        mcts_policy.visualize_best_path(mcts_policy.root_no_comm, ax)
        ax.axis("equal")

        plt.show()
    else:
        s_init = np.array([0.5, 0.5, 0.0])
        mcts_policy.generate_policy(s_init, t_max=0.8)

        fig, ax = plt.subplots()
        mcts_policy.visualize_search_tree(mcts_policy.policy_tree_root, ax)
        ax.axis("equal")

        plt.show()

        fig, ax = plt.subplots()
        mcts_policy.visualize_best_path(mcts_policy.policy_tree_root, ax)
        ax.axis("equal")

        plt.show()


if __name__ == "__main__":
    policy_search_example("/home/yuhang/Documents/proactive_guidance/training_data/user0/mdp_planenr_obs_haptic.pkl",
                          "haptic", flag_no_comm=True)
