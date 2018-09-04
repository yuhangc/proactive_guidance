#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle


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


class MovementModel(object):
    def __init__(self, modalities=None):
        self.x = 0.0
        self.y = 0.0
        self.alpha = 0.0

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

    def train(self, data_path):
        # fit parameters with data
        pass

    def save_param(self, root_path):
        for modality in self.modalities:
            model_file = root_path + "/movement_model_" + modality + ".pkl"
            with open(model_file, "w") as f:
                pickle.dump(self.params[modality], f)

    def sample_state(self, a, t):
        pass

    def sample_traj(self, a, T):
        pass

    # functions to sample/simulate each individual step
    def sample_delay(self, a):
        modality = a[0]
        delay_param = (self.params[modality].delay, self.params[modality].std_delay)
        return np.exp(np.random.normal(delay_param[0], delay_param[1], 1))

    def sample_turning(self, s, a, dt=None):
        pass

    def sample_walking(self, s, a, dt):
        pass


if __name__ == "__main__":
    pass
