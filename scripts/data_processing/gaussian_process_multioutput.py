#!/usr/bin/env python

import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, DotProduct
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


class VerySimpleModel(ModelBase):
    def __init__(self, comm, dt=0.5):
        super(VerySimpleModel, self).__init__(dt)

        self.comm = comm

    def create_kernel(self):
        return 1.0 * RBF(length_scale=2.0) + 1.0 * WhiteKernel(0.2)

    def load_data(self, data_file):
        with open(data_file) as f:
            t_all, pose_all, protocol_data = pickle.load(f)

        self.X = []
        self.y = []

        for t, pose, comm in zip(t_all, pose_all, protocol_data):
            if int(comm[0]) != self.comm[0]:
                continue

            if int(comm[1]) != self.comm[1]:
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

            # input (state) is defined as (pose(t), comm(t'), t-t')
            # output is defined as (pose(t+1))
            for i in range(len(t)):
                x = np.array([i])
                y = pose[i].copy()

                self.X.append(x)
                self.y.append(y)

    def sample_traj(self, T, n_samples=101):
        x = np.linspace(0, T, n_samples)
        y = self.gp.sample_y(x.reshape(-1, 1), random_state=None)

        return y.reshape(n_samples, -1)

    def predict_traj(self, T):
        x = np.arange(0, T)
        y = self.gp.predict(x.reshape(T, 1))

        return y.reshape(T, -1)


class TrajModel(ModelBase):
    def __init__(self, dt=0.5):
        super(TrajModel, self).__init__(dt)

    def create_kernel(self):
        return 1.0 * RBF(length_scale=[0.5, 0.5, 5.0]) + 0.2 * WhiteKernel(0.2) # + 1.0 * DotProduct(sigma_0=0.0)

    def load_data(self, data_file):
        with open(data_file) as f:
            t_all, pose_all, protocol_data = pickle.load(f)

        self.X = []
        self.y = []

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
            comm_vec = np.array([np.cos(comm[0]), np.sin(comm[0])]) * (comm[1] + 1)

            # input (state) is defined as (comm(t'), t-t')
            # output is defined as (pose(t+1))
            for i in range(len(t) - 1):
                x = np.array([comm_vec[0], comm_vec[1], i])
                y = pose[i+1].copy()

                self.X.append(x)
                self.y.append(y)

    def sample_traj(self, comm, T, n_samples=101):
        comm[0] *= np.pi * 0.25
        comm_vec = np.array([np.cos(comm[0]), np.sin(comm[0])]) * (comm[1] + 1)

        t = np.linspace(0, T, n_samples)
        x = np.hstack((np.tile(comm_vec, [n_samples, 1]), t.reshape(-1, 1)))

        y = self.gp.sample_y(x.reshape(n_samples, -1))

        return y.reshape(n_samples, -1)

        # for i in range(1, T):
        #     # print out information about the prediction
        #     state = np.array([comm_vec[0], comm_vec[1], i+1])
        #     pose = self.gp.sample_y(state.reshape(1, -1), random_state=None)[0].reshape(-1)
        #     traj.append(pose)

        # return np.asarray(traj)

    def predict_traj(self, comm, T):
        comm[0] *= np.pi * 0.25
        comm_vec = np.array([np.cos(comm[0]), np.sin(comm[0])]) * (comm[1] + 1)

        traj = []
        for i in range(T):
            state = np.array([comm_vec[0], comm_vec[1], i])
            pose, pose_std = self.gp.predict(state.reshape(1, -1), return_std=True)
            traj.append(pose[0].reshape(-1))

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


def very_simple_model_example(model_file, flag_train_model=False):
    if flag_train_model:
        simple_model = VerySimpleModel([0, 2], dt=0.5)

        simple_model.load_data("/home/yuhang/Documents/proactive_guidance/training_data/test0-0826/raw_transformed.pkl")

        print "Start to train model..."
        simple_model.train()
        print "Training finished"

        with open(model_file, "wb") as mf:
            pickle.dump(simple_model, mf)
    else:
        with open(model_file) as mf:
            simple_model = pickle.load(mf)
        print "model loaded"

    # sample a trajectories given different communication
    fig, axes = plt.subplots()

    for rep in range(5):
        traj = simple_model.sample_traj(10)
        axes.plot(traj[:, 0], traj[:, 1])
    axes.axis("equal")

    fig, axes = plt.subplots()
    traj = simple_model.predict_traj(10)
    axes.plot(traj[:, 0], traj[:, 1])
    axes.axis("equal")

    plt.show()


def traj_model_example(model_file, flag_train_model=False):
    if flag_train_model:
        simple_model = TrajModel()

        simple_model.load_data("/home/yuhang/Documents/proactive_guidance/training_data/test0-0826/raw_transformed.pkl")

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
    fig, axes = plt.subplots()

    for dir in range(0, 8):
        traj = simple_model.sample_traj([dir, 2], 10)
        axes.plot(traj[:, 0], traj[:, 1])
    axes.axis("equal")

    fig, axes = plt.subplots()

    for dir in range(0, 8):
        traj = simple_model.predict_traj([dir, 2], 10)
        axes.plot(traj[:, 0], traj[:, 1])
    axes.axis("equal")

    plt.show()


if __name__ == "__main__":
    # one_step_model_example("/home/yuhang/Documents/proactive_guidance/gp_models/test0-0820/one_step_model_multi.pkl",
    #                        True)

    # very_simple_model_example("/home/yuhang/Documents/proactive_guidance/"
    #                           "gp_models/test0-0826/very_simple_model_multi.pkl",
    #                           True)

    traj_model_example("/home/yuhang/Documents/proactive_guidance/gp_models/test0-0826/traj_model_multi.pkl",
                       False)
