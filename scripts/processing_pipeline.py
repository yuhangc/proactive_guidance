#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

import sys
sys.path.append("/home/yuhang/ros_dev/src/proactive_guidance/scripts/model_plan")
sys.path.append("/home/yuhang/ros_dev/src/proactive_guidance/scripts")

from data_processing.loading import load_save_all
from model_plan.pre_processing import pre_processing, pre_processing_no_orientation
from model_plan.gp_model_approx import model_approx_continuous_example, model_approx_create_interp_data
from model_plan.gp_model_approx import GPModelApproxBase
from model_plan.movement_model import save_default_model, MovementModel
from model_plan.policies import generate_mdp_policies, generate_naive_policies_from_mdp
from model_plan.policies import generate_policies_with_obs
from model_plan.simulation import Simulator


def process_user(root_path, usr, training_protocol, modalities):
    offsets = {"haptic": -0.25, "audio": -0.2}

    for modality in modalities:
        load_save_all(root_path + "/user" + str(usr) + "/" + modality,
                      training_protocol, flag_transform_to_body=True, rot_offset=offsets[modality])

        pre_processing(root_path, "user" + str(usr), modality)

        # load_save_all(root_path + "/user" + str(usr) + "/" + modality,
        #               training_protocol, flag_transform_to_body=False, rot_offset=offsets[modality])
        #
        # pre_processing_no_orientation(root_path, "user" + str(usr), modality)

        model_approx_continuous_example(root_path + "/user" + str(usr), modality, flag_train_model=True)

    model_approx_create_interp_data(root_path + "/user" + str(usr))


def train_unified_model(root_path, users):
    # load all pre-processed user data
    for modality in ["haptic", "audio"]:
        data_all = ()

        for user in users:
            with open(root_path + "/user" + str(user) + "/" + modality + "/processed.pkl") as f:
                data = pickle.load(f)

            data_all += (data, )

        data_all = np.vstack(data_all)

        with open(root_path + "/unified" + "/" + modality + "/processed.pkl", "w") as f:
            pickle.dump(data_all, f)

        # create a GP model
        model = GPModelApproxBase(dim=1)
        model.load_data(root_path + "/unified/" + modality)
        model.train()

        model.visualize_model()

        model.create_gp_interp_func(24)

        with open(root_path + "/unified/gp_model_" + modality + ".pkl", "w") as f:
            pickle.dump(model, f)

    # create and save movement model
    mmodel = MovementModel()
    mmodel.load_model(root_path + "/unified")
    mmodel.set_default_param()

    with open("../resources/pretrained_models/human_models/user_unified_default.pkl", "w") as f:
        pickle.dump(mmodel, f)


def gen_obs_policies(user, modality, flag_visulize_env=True, policy="mdp"):
    # 3 targets and 3 env settings
    n_targets = 3
    target_all = [[1.5, 4.5], [2.5, 3.5], [3.5, 2]]
    s_init = [-2.0, 0.5, 0.25 * np.pi]

    obs_all = []
    obs_all.append([[0.0, 3.25, 1.0, 2.75], [2.25, 3.25, 1.0, 2.75], [1.0, 5.25, 1.25, 0.75]])
    obs_all.append([[-1.0, 2.5, 2.0, 1.0], [0.0, 1.25, 1.5, 0.5]])
    obs_all.append([[0.0, 0.0, 0.75, 2.25], [1.5, 3.0, 2.0, 0.5], [2.0, 0.0, 1.0, 1.0]])

    # visualize the environment
    if flag_visulize_env:
        fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4))
        fig.tight_layout()

        for i in range(n_targets):
            for x, y, w, h in obs_all[i]:
                rect = Rectangle((x, y), w, h)
                axes[i].add_patch(rect)

            # axes[i].scatter(target_all[i][0], target_all[i][1], facecolor='r')
            rect = Rectangle((target_all[i][0], target_all[i][1]), 0.25, 0.25, facecolor='r', lw=0)
            axes[i].add_patch(rect)
            axes[i].scatter(s_init[0], s_init[1])

            axes[i].axis("equal")
            axes[i].set_xlim(-3, 4.5)
            axes[i].set_ylim(-1, 6)

        plt.show()

    generate_policies_with_obs(zip(target_all, obs_all),
                               "/home/yuhang/Documents/proactive_guidance/training_data",
                               modality, user, policy)


if __name__ == "__main__":
    data_path = "/home/yuhang/Documents/proactive_guidance/training_data"
    training_protocol = "../resources/protocols/random_continuous_protocol_5rep2.txt"
    modalities = ["haptic", "audio"]
    # modalities = ["audio"]

    user = 0

    save_path = "/home/yuhang/Documents/proactive_guidance/training_data/user" + str(user)

    # process_user(data_path, user, training_protocol, modalities)
    #
    # save_default_model(user, "../resources/pretrained_models/human_models")

    # generate_mdp_policies("../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                       save_path, "haptic", user)

    # generate_naive_policies_from_mdp(save_path + "/pretrained_model", "haptic")

    # model_approx_continuous_example(save_path, "haptic", False)

    # users = range(10
    # train_unified_model(data_path, users)

    # gen_obs_policies(user, "haptic")
    gen_obs_policies(user, "haptic", policy="mdp")
