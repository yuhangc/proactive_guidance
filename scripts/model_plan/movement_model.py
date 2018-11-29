#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle

from pre_processing import wrap_to_pi
from gp_model_approx import GPModelApproxBase


class MovementModelParam(object):
    def __init__(self):
        self.delay = 0.0
        self.om = 0.0
        self.vd = 0.0
        self.f_stray = 0.0

        self.kv = 0.0
        self.kalpha = 0.0

        self.std_delay = 0.0
        self.std_om = 0.0
        self.std_vd = 0.0
        self.std_vd_heading = 0.0


class MovementModel(object):
    def __init__(self, modalities=None):
        self.s = np.zeros((3, ))

        # gp models
        self.gp_model = {}

        # other parameters to model actual movements
        self.params = {}

        # feedback modalities
        if modalities is None:
            self.modalities = ["haptic", "audio"]

    def load_model(self, root_path, flag_load_param=False):
        # load gp model and parameters for haptic and audio feedback
        for modality in self.modalities:
            gp_file = root_path + "/gp_model_" + modality + ".pkl"
            with open(gp_file) as f:
                self.gp_model[modality] = pickle.load(f)

            if flag_load_param:
                model_file = root_path + "/movement_model_" + modality + ".pkl"
                with open(model_file) as f:
                    self.params[modality] = pickle.load(f)

    def set_default_param(self):
        self.params["haptic"] = MovementModelParam()
        self.params["audio"] = MovementModelParam()

        # self.params["haptic"].delay = np.log(1.0)
        # self.params["haptic"].om = 2.0
        self.params["haptic"].delay = np.log(1.1)
        self.params["haptic"].om = 1.6
        self.params["haptic"].vd = 0.5
        self.params["haptic"].std_delay = 0.1
        self.params["haptic"].std_om = 0.2
        # self.params["haptic"].std_vd = 0.05
        # self.params["haptic"].std_vd_heading = 0.05
        self.params["haptic"].std_vd = 0.1
        self.params["haptic"].std_vd_heading = 0.06

        self.params["audio"].delay = np.log(2.0)
        self.params["audio"].om = 2.0
        self.params["audio"].vd = 0.5
        self.params["audio"].std_delay = 0.1
        self.params["audio"].std_om = 0.2
        self.params["audio"].std_vd = 0.05
        self.params["audio"].std_vd_heading = 0.05

    def train(self, data_path):
        # fit parameters with data
        pass

    def save_param(self, root_path):
        for modality in self.modalities:
            model_file = root_path + "/movement_model_" + modality + ".pkl"
            with open(model_file, "w") as f:
                pickle.dump(self.params[modality], f)

    def set_state(self, x, y, alpha):
        self.s = np.array([x, y, alpha])

    def sample_state(self, a, dt, T, flag_check_stop=False, s_g=None):
        t, traj = self.sample_traj_single_action(a, dt, T)

        if flag_check_stop:
            # check each point in between
            for point in traj:
                d = np.linalg.norm(point[:2] - s_g[:2])
                if d < 0.20:
                    point[0] = s_g[0]
                    point[1] = s_g[1]
                    point[2] = wrap_to_pi(point[2])
                    return point
            traj[-1, 2] = wrap_to_pi(traj[-1, 2])
            return traj[-1]
        else:
            traj[-1, 2] = wrap_to_pi(traj[-1, 2])
            return traj[-1]

    def sample_traj_single_action(self, a, dt, T, flag_return_alp=False, flag_delay_move=True):
        """
        Simulate the trajectory given a single communication action
        :param a: action in the form of (modality, alpha_d)
        :param dt: desired sample time interval
        :param T: simulate until T
        :return: a list of time stamps and states
        """
        t_traj = [0.0]
        traj = [self.s.copy()]

        # check for stop
        if np.abs(a[1]) > 2.0 * np.pi:
            t_traj.append(T)
            traj.append(self.s.copy())
            return np.asarray(t_traj), np.asarray(traj)

        # first sample the delay
        t = self.sample_delay(a)

        if t < 0.6:
            t = 0.6
        elif t > 1.5:
            t = 1.5
        # if t >= T:
        #     # t_traj.append(T)
        #     # traj.append(self.s.copy())
        #     # return np.asarray(t_traj), np.asarray(traj)
        #     print "movement model: this should not happen...\r"
        #     t = T * 0.5

        if flag_delay_move:
            tt = 0.0
            s = self.s.copy()
            while tt + dt < t:
                tt += dt
                s += np.array([np.cos(self.s[2]), np.sin(self.s[2]), 0.0]) * dt * self.params["haptic"].vd * 0.4
                t_traj.append(tt)
                traj.append(s.copy())

            t_traj.append(t)
            s += np.array([np.cos(self.s[2]), np.sin(self.s[2]), 0.0]) * (t-tt) * self.params["haptic"].vd * 0.4
            traj.append(s.copy())
        else:
            s = self.s.copy()
            t_traj.append(t)
            traj.append(s.copy())

        # sample the "true" direction that human follows
        # ad_mean, ad_std = self.gp_model[a[0]].predict(a[1])[0]
        ad_mean, ad_std = self.gp_model[a[0]].predict_fast(a[1])[0]
        alpha_d = np.random.normal(ad_mean, ad_std) + self.s[2]

        # sample turning procedure
        s_next, dt_turn = self.sample_turning(s, (a[0], alpha_d), T - t)

        t += dt_turn
        t_traj.append(t)
        traj.append(s_next)

        if t >= T:
            if flag_return_alp:
                return alpha_d, np.asarray(t_traj), np.asarray(traj)
            else:
                return np.asarray(t_traj), np.asarray(traj)

        # sample straight walking motion
        s_new = s_next
        while t + dt < T:
            s_new = self.sample_walking(s_new, (a[0], alpha_d), dt)
            t += dt

            t_traj.append(t)
            traj.append(s_new)

        s_new = self.sample_walking(s_new, (a[0], alpha_d), T - t)
        t_traj.append(T)
        traj.append(s_new)

        if flag_return_alp:
            return alpha_d, np.asarray(t_traj), np.asarray(traj)
        else:
            return np.asarray(t_traj), np.asarray(traj)

    def sample_traj_no_action(self, a, dt, T):
        t_traj = [0.0]
        traj = [self.s.copy()]

        modality, alpha_d = a

        # sample straight walking motion
        t = 0.0
        s_new = traj[0]
        while t + dt < T:
            s_new = self.sample_walking(s_new, a, dt)
            t += dt

            t_traj.append(t)
            traj.append(s_new)

        s_new = self.sample_walking(s_new, (a[0], alpha_d), T - t)
        t_traj.append(T)
        traj.append(s_new)

        return np.asarray(t_traj), np.asarray(traj)

    def sample_traj_action_list(self, a_list, dt, T):
        pass

    # functions to sample/simulate each individual step
    def sample_delay(self, a):
        modality = a[0]
        delay_param = (self.params[modality].delay, self.params[modality].std_delay)
        return np.exp(np.random.normal(delay_param[0], delay_param[1]))

    def sample_turning(self, s, a, dt=None):
        # sample an angular velocity
        modality, alpha_d = a
        om_param = (self.params[modality].om, self.params[modality].std_om)
        om = np.random.normal(om_param[0], om_param[1])

        s_new = s.copy()
        dt_stop = np.abs(alpha_d - s[2]) / om

        if dt is None:
            s_new[2] = alpha_d
            return s_new, dt_stop
        else:
            if dt_stop >= dt:
                if alpha_d > s[2]:
                    s_new[2] += om * dt
                else:
                    s_new[2] -= om * dt
                dt_stop = dt
            else:
                s_new[2] = alpha_d

        return s_new, dt_stop

    def sample_walking(self, s, a, dt):
        # assume constant (but noisy) velocity and heading
        # dt should not be set too large
        modality, alpha_d = a
        v_param = (self.params[modality].vd, self.params[modality].std_vd)
        heading_param = (alpha_d, self.params[modality].std_vd_heading)

        v = np.random.normal(v_param[0], v_param[1])
        heading = np.random.normal(heading_param[0], heading_param[1])

        s_new = s.copy()
        s_new[0] += v * np.cos(heading) * dt
        s_new[1] += v * np.sin(heading) * dt
        s_new[2] = heading

        return s_new

    def get_prob_alp_d(self, a, alp_d):
        ad_mean, ad_std = self.gp_model[a[0]].predict_fast(a[1])[0]

        var = ad_std ** 2
        a_diff = alp_d - ad_mean

        return np.exp(-a_diff**2 / 2.0 / var) / np.sqrt(2.0 * np.pi * var)


def single_action_sample_example(n_samples, modality):
    # create a model object
    model = MovementModel()
    model.load_model("/home/yuhang/Documents/proactive_guidance/training_data/user0")
    model.set_default_param()

    # sample with a few actions
    T = 6.0
    n_dir = 8
    alphas = wrap_to_pi(np.pi * 2.0 / n_dir * np.arange(0, n_dir))

    all_traj = []
    for i in range(n_dir):
        traj_dir = []
        for rep in range(n_samples):
            traj_dir.append(model.sample_traj_single_action((modality, alphas[i]), 0.5, T))

        all_traj.append(traj_dir)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i in range(2):
        for j in range(4):
            for t, traj in all_traj[i*4+j]:
                axes[i][j].plot(t, np.asarray(traj)[:, 2])

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i in range(2):
        for j in range(4):
            for t, traj in all_traj[i*4+j]:
                d = np.linalg.norm(np.asarray(traj)[:, :2], axis=1)
                axes[i][j].plot(t, d)

    plt.show()


def save_default_model(usr_id, save_path):
    # create a model object
    model = MovementModel()
    model.load_model("/home/yuhang/Documents/proactive_guidance/training_data/user" + str(usr_id))
    model.set_default_param()

    # save model
    with open(save_path + "/user" + str(usr_id) + "_default.pkl", "w") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    # single_action_sample_example(5, "audio")
    save_default_model(0, "../../resources/pretrained_models/human_models")
