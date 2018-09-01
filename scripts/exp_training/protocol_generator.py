#!/usr/bin/env python
import numpy as np


def generate_naive_random_policy(file_name, n_reps, n_dir, n_mag):
    data = []

    for dir in range(n_dir):
        for mag in range(n_mag):
            data.append([dir, mag])

    # repeat and randomly permute data
    data = np.asarray(data)
    data = np.tile(data, (n_reps, 1))
    np.random.shuffle(data)

    np.savetxt(file_name, data, fmt="%2d", delimiter=", ")


def generate_naive_continuous_random_policy(file_name, n_dir, n_rep, mag):
    data = []

    angles = np.linspace(0, 360 - 360 / n_dir, n_dir)

    for i in range(n_rep):
        for ang in angles:
            data.append([ang, mag])

    # repeat and randomly permute data
    data = np.asarray(data, dtype=int)
    np.random.shuffle(data)

    np.savetxt(file_name, data, fmt="%d", delimiter=", ")


def generate_naive_policy(file_name, n_trials, x_range, y_range):
    pass


if __name__ == "__main__":
    # generate_naive_random_policy("random_protocol.txt", 5, 8, 3)
    generate_naive_continuous_random_policy("random_continuous_protocol_10rep2.txt", 24, 10, 2)
