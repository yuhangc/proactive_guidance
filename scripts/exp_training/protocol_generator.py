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
    start_pose = np.array([-1.0, 3.0, 1.0])

    # target_dirs = np.deg2rad(np.array([30.0, -30.0, 150.0, -150.0]))
    target_dirs = np.deg2rad(np.array([30.0, -30.0]))
    confusion_ang = np.deg2rad(15.0)
    target_dist = 5.0
    confusion_dist = 4.0

    targets = []

    target_id = -1
    for ang in target_dirs:
        # targets are in robot/laser reference frame
        target_id += 1
        pos = np.zeros((4, ))
        pos[0] = target_id
        pos[2:] = np.array([np.cos(ang), np.sin(ang)]) * target_dist + start_pose[:2] - robot_pose[:2]

        for i in range(n_rep_trial):
            targets.append(pos)

        for alp in [-confusion_ang, confusion_ang]:
            ang1 = ang + alp

            target_id += 1
            pos = np.zeros((4, ))
            pos[0] = target_id
            pos[2:] = np.array([np.cos(ang1), np.sin(ang1)]) * confusion_dist + start_pose[:2] - robot_pose[:2]

            for i in range(n_rep_throw):
                targets.append(pos)

    # randomly shuffle targets
    targets = np.asarray(targets)
    np.random.shuffle(targets)

    np.savetxt(file_name, targets, fmt="%.2f", delimiter=", ")


def generate_free_space_protocol2(file_name, n_policies=2):
    # set parameters
    robot_pose = np.array([0.0, 1.0, 0.0])
    start_pose = np.array([-1.0, 3.0, 1.0])

    # target_dirs = np.deg2rad(np.array([30.0, -30.0, 150.0, -150.0]))
    # target directions, magnitudes and repetitions
    target_dirs = np.deg2rad(np.array([0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0]))
    target_mag = np.array([4.5, 4.5, 4.5, 4.0, 4.0, 3.5, 3.5])
    # target_rep = np.array([3, 3, 3, 3, 3, 3, 3])
    target_rep = np.array([1, 1, 1, 1, 1, 1, 1])

    targets = []

    for i in range(len(target_dirs)):
        # targets are in robot/laser reference frame
        ang = target_dirs[i]

        trial_data = np.zeros((4, ))
        trial_data[0] = i
        trial_data[2:] = np.array([np.cos(ang), np.sin(ang)]) * target_mag[i] + start_pose[:2] - robot_pose[:2]

        for k in range(target_rep[i]):
            for p in range(n_policies):
                trial_data[1] = p
                targets.append(trial_data.copy())

    # randomly shuffle targets
    targets = np.asarray(targets)
    np.random.shuffle(targets)

    np.savetxt(file_name, targets, fmt="%.2f", delimiter=", ")


if __name__ == "__main__":
    # generate_naive_random_policy("random_protocol.txt", 5, 8, 3)
    # generate_naive_continuous_random_protocol("random_continuous_protocol_10rep2.txt", 24, 10, 2)
    # generate_free_space_protocol("free_space_exp_protocol_2targets.txt", 5, 2)
    generate_free_space_protocol2("free_space_exp_protocol_7targets_mixed_train.txt", n_policies=3)

