#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append("/home/yuhang/ros_dev/src/proactive_guidance/scripts/model_plan")
sys.path.append("/home/yuhang/ros_dev/src/proactive_guidance/scripts")

from data_processing.loading import load_save_all
from model_plan.pre_processing import pre_processing, pre_processing_no_orientation
from model_plan.gp_model_approx import model_approx_continuous_example, model_approx_create_interp_data
from model_plan.movement_model import save_default_model
from model_plan.policies import generate_mdp_policies, generate_naive_policies_from_mdp
from model_plan.simulation import Simulator


def process_user(root_path, usr, training_protocol, modalities):
    offsets = {"haptic": -0.25, "audio": -0.35}

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


if __name__ == "__main__":
    data_path = "/home/yuhang/Documents/proactive_guidance/training_data"
    training_protocol = "../resources/protocols/random_continuous_protocol_5rep2.txt"
    modalities = ["haptic", "audio"]
    # modalities = ["audio"]

    user = 7
    save_path = "/home/yuhang/Documents/proactive_guidance/training_data/user" + str(user)

    # process_user(data_path, 6, training_protocol, modalities)
    #
    # save_default_model(6, "../resources/pretrained_models/human_models")

    # generate_mdp_policies("../resources/protocols/free_space_exp_protocol_7targets_mdp.txt",
    #                       save_path, "haptic", user)

    generate_naive_policies_from_mdp(save_path + "/pretrained_model", "haptic")

    # model_approx_continuous_example("/home/yuhang/Documents/proactive_guidance/training_data/user1",
    #                                 "audio", False)
