#!/usr/bin/env python
import numpy as np


def generate_naive_random_protocol(file_name, n_reps, n_dir, n_mag):
    data = []

    for dir in range(n_dir):
        for mag in range(n_mag):
            data.append([dir, mag])

    # repeat and randomly permute data
    data = np.asarray(data)
    data = np.tile(data, (n_reps, 1))
    np.random.shuffle(data)

    np.savetxt(file_name, data, fmt="%2d", delimiter=", ")


def generate_naive_continuous_random_protocol(file_name, n_dir, n_rep, mag):
    data = []

    angles = np.linspace(0, 360 - 360 / n_dir, n_dir)

    for i in range(n_rep):
        for ang in angles:
            data.append([ang, mag])

    # repeat and randomly permute data
    data = np.asarray(data, dtype=int)
    np.random.shuffle(data)

    np.savetxt(file_name, data, fmt="%d", delimiter=", ")


def generate_free_space_protocol(file_name, n_rep_trial, n_rep_throw):
    # set parameters
    robot_pose = np.array([0.0, 1.0, 0.0])
    start_pose = np.array([0.0, 3.0, 1.0])

    target_dirs = np.deg2rad(np.array([30.0, -30.0, 150.0, -150.0]))
    # target_dirs = np.deg2rad(np.array([30.0, -30.0]))
    confusion_ang = np.deg2rad(15.0)
    target_dist = 4.5

    targets = []

    for ang in target_dirs:
        # targets are in robot/laser reference frame
        pos = np.zeros((3, ))
        pos[1:] = np.array([np.cos(ang), np.sin(ang)]) * target_dist + start_pose[:2] - robot_pose[:2]

        for i in range(n_rep_trial):
            targets.append(pos)

        for alp in [-confusion_ang, confusion_ang]:
            ang1 = ang + alp
            pos = np.zeros((3, ))
            pos[1:] = np.array([np.cos(ang1), np.sin(ang1)]) * target_dist + start_pose[:2] - robot_pose[:2]

            for i in range(n_rep_throw):
                targets.append(pos)

    # randomly shuffle targets
    targets = np.asarray(targets)
    np.random.shuffle(targets)

    np.savetxt(file_name, targets, fmt="%.2f", delimiter=", ")


if __name__ == "__main__":
    # generate_naive_random_policy("random_protocol.txt", 5, 8, 3)
    # generate_naive_continuous_random_protocol("random_continuous_protocol_10rep2.txt", 24, 10, 2)
    generate_free_space_protocol("free_space_exp_protocol_4targets.txt", 5, 2)

