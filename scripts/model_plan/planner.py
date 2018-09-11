#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import pickle

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
    def __init__(self, s, belief):
        super(MCTSStateNode, self).__init__("state")

        self.s = s
        self.alp_d_mean, self.alp_d_std = belief

        self.a_list = []
        self.n_a = 0


class MCTSActionNode(MCTSNodeBase):
    def __init__(self, s, a):
        super(MCTSActionNode, self).__init__("action")

        self.a = a
        self.s = s


class MCTSTrajNode(MCTSNodeBase):
    def __init__(self, alp_d, traj):
        super(MCTSTrajNode, self).__init__("traj")

        self.alp_d = alp_d
        self.traj = traj
        self.selected = np.zeros_like(traj[0], dtype=int)


class MCTSPolicy:
    def __init__(self, model, default_policy, modality):
        self.policy_tree = None

        self.model = model
        self.default_policy = default_policy
        self.modality = modality

        # other settings
        self.n_actions_per_state = 5
        self.da_sample = np.deg2rad(10)

        self.action_widen_factor = 0.5
        self.traj_widen_factor = 0.5
        self.time_widen_factor = 0.5

        self.uct_weight = 1.0

        self.traj_sample_dt = 0.5
        self.traj_sample_T = 5.0

        self.time_sample_factor1 = 1.0
        self.time_sample_factor2 = 1.0

        self.gamma = 0.95
        self.gamma_scale = 0.5

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
        # generate sampling list if none
        if not state_node.a_list:
            state_node.a_list = self.gen_action_list(state_node.s)

        # decide to sample a new action or sample from existing ones
        if state_node.n_a < len(state_node.a_list) and\
                        np.power(state_node.count, self.action_widen_factor) > state_node.n_a:
            a = state_node.a_list[state_node.n_a]
            state_node.n_a += 1

            action_node = MCTSActionNode(state_node.s, a)
            state_node.children.append(action_node)
        else:
            # compute uct scores for all actions
            child_id = self.select_by_uct(state_node)
            action_node = state_node.children[child_id]

        return action_node

    def select_traj(self, action_node):
        # decide to sample a new trajectory or use existing ones
        n = len(action_node.children)
        if np.power(action_node.count, self.traj_widen_factor) > n:
            self.model.set_state(action_node.s[0], action_node.s[1], action_node.s[2])
            alp_d, t, traj = self.model.sample_traj_single_action((self.modality, action_node.a),
                                                                  self.traj_sample_dt,
                                                                  self.traj_sample_T,
                                                                  flag_return_alp=True)

            traj_node = MCTSTrajNode(alp_d, (t[2:], traj[2:]))
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

        for i in range(len(traj_node.traj[0])):
            if (traj_node.selected[i] > 0) ^ flag_new_node:
                v = self.default_policy.get_value_func(traj_node.traj[1][i])
                alp_d = self.default_policy.sample_policy(traj_node.traj[1][i])
                alp_diff = wrap_to_pi(traj_node.alp_d - alp_d)

                val = self.time_sample_factor1 * v * self.gamma**(traj_node.traj[0][i] * self.gamma_scale) +\
                      self.time_sample_factor2 * alp_diff**2

                elm.append(i)
                chances.append(np.exp(val))

        chances = np.asarray(chances) / np.sum(chances)

        if flag_new_node:
            return np.random.choice(elm, p=chances)
        else:
            return np.random.choice(np.arange(len(traj_node.children)), p=chances)

    def select_time(self, traj_node):
        n = len(traj_node.children)
        if np.power(traj_node.count, self.time_widen_factor) > n:
            i = self.sample_time(traj_node, True)
            state_node = MCTSStateNode(traj_node.traj[1][i], (traj_node.alp_d, 0.0))
            traj_node.children.append(state_node)
        else:
            i = self.sample_time(traj_node, False)
            state_node = traj_node.children[i]

        return state_node


if __name__ == "__main__":
    pass
