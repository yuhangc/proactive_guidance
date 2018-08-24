#!/usr/bin/env python

import pickle
import numpy as np

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class ModelBase(object):
    def __init__(self):
        self.X = None
        self.y = None

        self.gp = gaussian_process.GaussianProcessRegressor(kernel=self.create_kernel())

    def create_kernel(self):
        raise IndexError("Method must be implemented!")

    def load_data(self, data_file):
        raise Exception("Method must be implemented!")

    def train(self):
        self.gp.fit(self.X, self.y)

    def predict(self):
        pass

    def sample(self, N):
        pass


class OneStepModel(ModelBase):
    def __init__(self):
        super(OneStepModel, self).__init__()

    def create_kernel(self):
        pass

    def load_data(self, data_file):
        with open(data_file) as f:
            t_all, pose_all, protocol_data = pickle.load(f)

        self.X = []
        self.y = []

        for t, pose, comm in zip(t_all, pose_all, protocol_data):
            # input (state) is defined as (pose(t), comm(t'), pose(t'), t-t')
            # output is defined as (pose(t+1))
            for i in range(len(t) - 1):
                x = [pose[i, 0], pose[i, 1], pose[i, 2], comm[0], comm[1],
                     pose[0, 0], pose[0, 1], pose[0, 2], i]
