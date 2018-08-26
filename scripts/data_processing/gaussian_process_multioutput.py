#!/usr/bin/env python

import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from loading import wrap_to_pi


class ModelBase(object):
    def __init__(self, dt=0.5):
        self.X = None
        self.y = None

        self.dt = dt

        self.gp = gaussian_process.GaussianProcessRegressor(kernel=self.create_kernel())

    def create_kernel(self):
        raise IndexError("Method must be implemented!")

    def load_data(self, data_file):
        raise Exception("Method must be implemented!")

    def train(self):
        self.gp.fit(self.X, self.y)

    def predict(self, x):
        return self.gp.predict(x, return_cov=True)

    def sample(self, x, n):
        return self.gp.sample_y(x, n_samples=n)


class OneStepModel(ModelBase):
    def __init__(self, dt=0.5):
        super(OneStepModel, self).__init__(dt)

    def create_kernel(self):
        return 1.0 * RBF(length_scale=[0.5, 0.5, 0.5, 0.2, 0.2, 0.5, 0.5]) + 1.0 * WhiteKernel(0.2)

    def load_data(self, data_file):
        with open(data_file) as f:
            t_all, pose_all, protocol_data = pickle.load(f)

        self.X = []
        self.y = []

        for t, pose, comm in zip(t_all, pose_all, protocol_data):
            if int(comm[0]) != 0:
                continue

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
                self.y.append(y)

    def sample_traj(self, pose_init, comm, T):
        comm[0] *= np.pi * 0.25

        traj = [pose_init.copy()]
        state = np.array([pose_init[0], pose_init[1], pose_init[2],
                          np.cos(comm[0]), np.sin(comm[0]), comm[1], 0])

        for i in range(T-1):
            # print out information about the prediction
            print state.reshape(1, -1)
            y, y_cov = self.gp.predict(state.reshape(1, -1), return_cov=True)
            print "at t = ", i, "mean is: ", y, "cov is:"
            print y_cov

            pose_next = self.gp.sample_y(state.reshape(1, -1))[0].reshape(-1)
            traj.append(pose_next)

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

    for dir in range(0, 1):
        traj = simple_model.sample_traj(pose_init, [dir, 0], 10)
        axes.plot(traj[:, 0], traj[:, 1])

    plt.show()


if __name__ == "__main__":
    one_step_model_example("/home/yuhang/Documents/proactive_guidance/gp_models/test0-0820/one_step_model_multi.pkl",
                           True)
