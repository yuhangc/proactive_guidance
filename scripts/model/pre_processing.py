#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle


def wrap_to_pi(ang):
    return (ang + np.pi) % (2 * np.pi) - np.pi


def pre_processing(root_path, usr, cond, flag_visualize=True):
    # first process the one-step case
    # load the preprocessed data
    path = root_path + "/user" + str(usr) + "/" + cond
    data_file = path + "/raw_transformed.pkl"

    with open(data_file) as f:
        t_all, pose_all, protocol_data = pickle.load(f)

    # compute final travel distance and and directions
    dir_out = []
    mag_out = []
    th_out = []
    for pose in pose_all:
        # average the last 10 samples
        pos_avg = np.mean(pose[-10:, 0:2], axis=0)
        dir_out.append(np.arctan2(pos_avg[1], pos_avg[0]))
        mag_out.append(np.linalg.norm(pos_avg))
        th_out.append(wrap_to_pi(np.mean(pose[-10:, 2])))

    dir_in = protocol_data[:, 0] * np.pi / 180.0 - np.pi * 0.5

    dir_out = wrap_to_pi(np.asarray(dir_out))
    dir_in = wrap_to_pi(dir_in)
    mag_out = np.asarray(mag_out)

    # wrap the dir_out around at end points (-pi and pi)
    for i in range(len(dir_out)):
        if dir_out[i] - dir_in[i] > np.pi:
            dir_out[i] -= 2.0 * np.pi
        elif dir_out[i] - dir_in[i] < -np.pi:
            dir_out[i] += 2.0 * np.pi

    # create 2 different plots
    if flag_visualize:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].scatter(np.rad2deg(dir_in), np.rad2deg(dir_out))
        axes[1].scatter(np.rad2deg(dir_in), mag_out)

        plt.show()

    # save data
    with open(path + "/processed.pkl", 'w') as f:
        pickle.dump(np.hstack((dir_in.reshape(-1, 1), dir_out.reshape(-1, 1), mag_out.reshape(-1, 1))), f)


if __name__ == "__main__":
    # pre_processing("/home/yuhang/Documents/proactive_guidance/training_data", 0, "one_step")
    # pre_processing("/home/yuhang/Documents/proactive_guidance/training_data", 0, "continuous")
    pre_processing("/home/yuhang/Documents/proactive_guidance/training_data", 0, "audio")
