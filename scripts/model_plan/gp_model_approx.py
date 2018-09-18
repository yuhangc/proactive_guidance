#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel, ConstantKernel


# utility function for outlier rejection
def outlier_rejection(x, m=2.0):
    return x[np.abs(x - np.mean(x)) < m * np.std(x)]


class GPModelApproxBase(object):
    def __init__(self, dim):
        # fit a 1-d gaussian process for each dimension
        self.dim = dim

        # training data
        self.X = None
        self.y_mean = None
        self.y_std = None
        self.data = None

        # gaussian process objects
        self.gp_mean, self.gp_std = self.create_gp()

        # samples used for fast interpolation
        self.gp_x = None
        self.gp_mean_f = None
        self.gp_std_f = None

        self.n_samples_fast_prediction = 24

    def create_gp(self):
        gp_mean = []
        gp_std = []

        for i in range(self.dim):
            # gp_mean.append(gaussian_process.GaussianProcessRegressor(kernel=1.0 * RBF(length_scale=1.0) +
            #                                                                 ConstantKernel(0.02, constant_value_bounds=(0.01, 0.05)) * \
            #                                                                 WhiteKernel(0.1, noise_level_bounds=(0.01, 0.2)),
            #                                                          n_restarts_optimizer=3))
            gp_mean.append(gaussian_process.GaussianProcessRegressor(kernel=1.0 * RBF(length_scale=1.0) +
                                                                            0.1 *
                                                                            WhiteKernel(0.1, noise_level_bounds=(0.01, 0.2)),
                                                                     n_restarts_optimizer=3))

            # gp_std.append(gaussian_process.GaussianProcessRegressor(kernel=1.0 * RBF(length_scale=0.3) +
            #                                                                0.01 * WhiteKernel(0.01),
            #                                                         n_restarts_optimizer=3))

            kernel_std = 1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                                              length_scale_bounds=(0.5, 5.0),
                                              periodicity_bounds=(1.0, 10.0)) + \
                         ConstantKernel(0.1, constant_value_bounds=(0.05, 0.2)) * \
                         WhiteKernel(0.01, noise_level_bounds=(0.01, 0.1))
            gp_std.append(gaussian_process.GaussianProcessRegressor(kernel=kernel_std,
                                                                    n_restarts_optimizer=3))
        return gp_mean, gp_std

    def create_gp_interp_func(self, n_samples):
        self.gp_x = np.arange(0, n_samples) * 2.0 * np.pi / n_samples - np.pi
        self.gp_mean_f = []
        self.gp_std_f = []

        for i in range(self.dim):
            self.gp_mean_f.append(self.gp_mean[i].predict(self.gp_x.reshape(-1, 1)).reshape(-1))
            self.gp_std_f.append(self.gp_std[i].predict(self.gp_x.reshape(-1, 1)).reshape(-1))

    def load_data(self, root_path, flag_outlier_rejection=True):
        with open(root_path + "/processed.pkl") as f:
            self.data = pickle.load(f)

        input_mapping = {}
        for data_point in self.data:
            x = data_point[0]

            if input_mapping.has_key(x):
                input_mapping[x].append(data_point[1:])
            else:
                input_mapping[x] = [data_point[1:]]

        self.X = np.asarray(input_mapping.keys())
        self.y_mean = [[] for i in range(self.dim)]
        self.y_std = [[] for i in range(self.dim)]

        for key, value in input_mapping.items():
            meas = np.asarray(value)
            for i in range(self.dim):
                if flag_outlier_rejection:
                    inliers = outlier_rejection(meas[:, i])
                else:
                    inliers = meas[:, 1]

                self.y_mean[i].append(np.mean(inliers))
                self.y_std[i].append(np.std(inliers))

    def train(self):
        for i in range(self.dim):
            self.gp_mean[i].fit(self.X.reshape(-1, 1), self.y_mean[i])
            self.gp_std[i].fit(self.X.reshape(-1, 1), self.y_std[i])

    def predict(self, dir_in):
        x = np.array([dir_in]).reshape(-1, 1)
        y = []

        for i in range(self.dim):
            y.append((self.gp_mean[i].predict(x), self.gp_std[i].predict(x)))

        return y

    def predict_fast(self, dir_in):
        y = []
        for i in range(self.dim):
            y.append((np.interp(dir_in, self.gp_x, self.gp_mean_f[i]),
                      np.interp(dir_in, self.gp_x, self.gp_std_f[i])))

        return y

    @staticmethod
    def _visualize_gp(gp, ax, x_train, y_train, xlabel, ylabel):
        n_samples = 50
        x = np.linspace(-np.pi, np.pi, n_samples)

        y_mean, y_std = gp.predict(x.reshape(n_samples, -1), return_std=True)

        ax.plot(x, y_mean, 'k', lw=3, zorder=9)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                        alpha=0.2, color='k')

        # y_samples = gp.sample_y(x.reshape(n_samples, -1), 5)
        # ax.plot(x, y_samples, lw=1)

        ax.scatter(x_train, y_train, c='r', s=10, zorder=10, edgecolors=(0, 0, 0))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def _visualize_process(self, dim, ax, xlabel, ylabel):
        n_samples = 50
        x = np.linspace(-np.pi, np.pi, n_samples)

        y_mean = self.gp_mean[dim].predict(x.reshape(n_samples, -1))
        y_std = self.gp_std[dim].predict(x.reshape(n_samples, -1))

        ax.plot(x, y_mean, 'k', lw=3, zorder=9)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                        alpha=0.2, color='k')

        ax.scatter(self.data[:, 0], self.data[:, dim+1], c='r', s=10, zorder=10, edgecolors=(0, 0, 0))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def visualize_model(self):
        # visualize each individual gp
        fig, axes = plt.subplots(2, self.dim, figsize=(5*self.dim, 5))

        if self.dim > 1:
            for i in range(self.dim):
                self._visualize_gp(self.gp_mean[i], axes[0, i], self.X, self.y_mean[i],
                                   "feedback(rad)", "response mean(rad)")
                self._visualize_gp(self.gp_std[i], axes[1, i], self.X, self.y_std[i],
                                   "feedback(rad)", "response std(rad)")
        else:
            self._visualize_gp(self.gp_mean[0], axes[0], self.X, self.y_mean[0],
                               "feedback(rad)", "response mean(rad)")
            self._visualize_gp(self.gp_std[0], axes[1], self.X, self.y_std[0],
                               "feedback(rad)", "response std(rad)")

        # visualize each dimension of the model
        fig, axes = plt.subplots(self.dim, 1, figsize=(5, 5))

        if self.dim > 1:
            for i in range(self.dim):
                self._visualize_process(i, axes[i], "feedback(rad)", "response data(rad)")
        else:
            self._visualize_process(0, axes, "feedback(rad)", "response data(rad)")

        # plot the "perfect" response
        x = [-np.pi, np.pi]
        axes.plot(x, x, 'r')

        plt.show()


def model_approx_one_step_example(root_path, flag_train_model=True):
    if flag_train_model:
        # train a one step model
        model_one_step = GPModelApproxBase(dim=2)

        model_one_step.load_data(root_path + "/one_step")
        model_one_step.train()
    else:
        with open(root_path + "/one_step_model.pkl") as f:
            model_one_step = pickle.load(f)

    model_one_step.visualize_model()

    if flag_train_model:
        with open(root_path + "/one_step_model.pkl", "w") as f:
            pickle.dump(model_one_step, f)


def model_approx_continuous_example(root_path, modality, flag_train_model=True):
    if flag_train_model:
        # train a one step model
        model_continuous = GPModelApproxBase(dim=1)

        model_continuous.load_data(root_path + "/" + modality)
        model_continuous.train()
    else:
        with open(root_path + "/gp_model_" + modality + ".pkl") as f:
            model_continuous = pickle.load(f)

    model_continuous.visualize_model()

    if flag_train_model:
        with open(root_path + "/gp_model_" + modality + ".pkl", "w") as f:
            pickle.dump(model_continuous, f)


def model_approx_create_interp_data(root_path):
    with open(root_path + "/gp_model_haptic.pkl") as f:
        model_haptic = pickle.load(f)

    with open(root_path + "/gp_model_audio.pkl") as f:
        model_audio = pickle.load(f)

    model_haptic.create_gp_interp_func(24)
    model_audio.create_gp_interp_func(24)

    with open(root_path + "/gp_model_haptic.pkl", "w") as f:
        pickle.dump(model_haptic, f)

    with open(root_path + "/gp_model_audio.pkl", "w") as f:
        pickle.dump(model_audio, f)


if __name__ == "__main__":
    # model_approx_one_step_example("/home/yuhang/Documents/proactive_guidance/training_data/user0", False)
    # model_approx_continuous_example("/home/yuhang/Documents/proactive_guidance/training_data/user0",
    #                                 "continuous", False)
    # model_approx_continuous_example("/home/yuhang/Documents/proactive_guidance/training_data/user1",
    #                                 "audio", False)
    model_approx_create_interp_data("/home/yuhang/Documents/proactive_guidance/training_data/user1")
