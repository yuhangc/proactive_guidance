#!/usr/bin/env python

import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from loading import wrap_to_pi


class ModelBase(object):
    def __init__(self, dt=0.5, dim=3):
        self.X = None
        self.y = None

        self.dt = dt
        self.dim = dim

        self.gp = []
        for i in range(dim):
            self.gp.append(gaussian_process.GaussianProcessRegressor(kernel=self.create_kernel(),
                                                                     n_restarts_optimizer=3))

    def create_kernel(self):
        raise IndexError("Method must be implemented!")

    def load_data(self, data_file):
        raise Exception("Method must be implemented!")

    def train(self):
        for gp, y in zip(self.gp, self.y):
            gp.fit(self.X, y)

    def predict(self, x):
        y = []
        y_std = []

        for gp in self.gp:
            yi, stdi = gp.predict(x.reshape(1, -1), return_std=True)
            y.append(yi)
            y_std.append(stdi)

        return np.asarray(y), np.asarray(y_std)

    def sample(self, x, n=1):
        y = []

        for gp in self.gp:
            y.append(gp.sample_y(x.reshape(1, -1), n_samples=n))

        return np.asarray(y)


class OneStepModel(ModelBase):
    def __init__(self, dt=0.5):
        super(OneStepModel, self).__init__(dt, dim=3)

    def create_kernel(self):
        return 1.0 * RBF(length_scale=[0.5, 0.5, 0.5, 0.2, 0.2, 0.5, 0.5]) + 1.0 * WhiteKernel(0.2)

    def load_data(self, data_file):
        with open(data_file) as f:
            t_all, pose_all, protocol_data = pickle.load(f)

        self.X = []
        self.y = [[], [], []]

        for t, pose, comm in zip(t_all, pose_all, protocol_data):
            # first down-sample the data
            step = int(self.dt / 0.025)
            t = t[::step]
            pose = pose[::step, :]

            # transform data w.r.t to the initial pose
            # assuming that the initial pose is always aligned with world frame
            # otherwise the position needs to be rotated
            pose -= pose[0]

            # convert angles to rad and wrap to pi
            pose[:, 2] = np.deg2rad(pose[:, 2])
            pose[:, 2] = wrap_to_pi(pose[:, 2])

            # convert communication direction to ang
            comm[0] *= np.pi * 0.25
            comm[0] = wrap_to_pi(comm[0])

            # input (state) is defined as (pose(t), comm(t'), t-t')
            # output is defined as (pose(t+1))
            for i in range(len(t) - 1):
                x = np.array([pose[i, 0], pose[i, 1], pose[i, 2],
                              np.cos(comm[0]), np.sin(comm[0]), comm[1], i])
                y = pose[i+1].copy()

                self.X.append(x)

                for dim in range(self.dim):
                    self.y[dim].append(y[dim])

    def sample_traj(self, pose_init, comm, T):
        comm[0] *= np.pi * 0.25

        traj = [pose_init.copy()]
        state = np.array([pose_init[0], pose_init[1], pose_init[2],
                          np.cos(comm[0]), np.sin(comm[0]), comm[1], 0])

        for i in range(T-1):
            # print out information about the prediction
            # print state.reshape(1, -1)
            # y, y_cov = self.predict(state)
            # print "at t = ", i, "mean is: ", y, "cov is:"
            # print y_cov

            pose_next = self.sample(state)
            traj.append(pose_next.reshape(-1))

            state = np.array([pose_next[0], pose_next[1], pose_next[2],
                              np.cos(comm[0]), np.sin(comm[0]), comm[1], i+1])

        return np.asarray(traj)


def one_step_model_example(model_file, flag_train_model=False):
    if flag_train_model:
        simple_model = OneStepModel()

        simple_model.load_data("/home/yuhang/Documents/proactive_guidance/training_data/test0-0820/raw_transformed.pkl")

        print "Start to train model..."
        simple_model.train()
        print "Training finished"
        print simple_model.gp

        with open(model_file, "wb") as mf:
            pickle.dump(simple_model, mf)
    else:
        with open(model_file) as mf:
            simple_model = pickle.load(mf)
        print "model loaded"

    # sample a trajectories given different communication
    pose_init = np.array([0.0, 0.0, 0.0])
    fig, axes = plt.subplots()

    for dir in range(8):
        traj = simple_model.sample_traj(pose_init, [dir, 2], 10)
        axes.plot(traj[:, 0], traj[:, 1])

    plt.show()


if __name__ == "__main__":
    one_step_model_example("/home/yuhang/Documents/proactive_guidance/gp_models/test0-0820/one_step_model.pkl", False)
