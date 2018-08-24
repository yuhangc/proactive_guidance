#!/usr/bin/env python

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
