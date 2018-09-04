#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from pre_processing import wrap_to_pi


class NaivePolicy(object):
    def __init__(self):
        self.s_g = None

    def set_target(self, s_g):
        self.s_g = s_g

    def compute_policy(self, s):
        # directly point to the direction of goal
        alpha_d = np.arctan2(self.s_g[1] - s[1], self.s_g[0] - s[0])
        alpha_d = wrap_to_pi(alpha_d - s[2])

        return alpha_d


if __name__ == "__main__":
    pass
